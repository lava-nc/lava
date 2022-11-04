// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/


#include <message_infrastructure/csrc/channel/dds/fast_dds.h>
#include <message_infrastructure/csrc/core/message_infrastructure_logging.h>

#include <fastdds/dds/domain/DomainParticipantFactory.hpp>

#include <fastdds/rtps/transport/shared_mem/SharedMemTransportDescriptor.h>
#include <fastdds/rtps/transport/TCPv4TransportDescriptor.h>
#include <fastdds/rtps/transport/TCPv6TransportDescriptor.h>
#include <fastdds/rtps/transport/UDPv4TransportDescriptor.h>
#include <fastdds/rtps/transport/UDPv6TransportDescriptor.h>
#include <fastrtps/Domain.h>

namespace message_infrastructure {

using namespace eprosima::fastdds::dds;  // NOLINT
using namespace eprosima::fastdds::rtps;  // NOLINT
using namespace eprosima::fastrtps::rtps;  // NOLINT

FastDDSPublisher::~FastDDSPublisher() {
  LAVA_LOG(LOG_DDS, "FastDDS Publisher releasing...\n");
  if (listener_->matched_ > 0) {
    LAVA_LOG_ERR("Still %d DataReader Listen\n", listener_->matched_);
  }
  if (!stop_) {
    LAVA_LOG_WARN(LOG_DDS, "Please stop Publisher before release it next time\n");
    Stop();
  }
  LAVA_DEBUG(LOG_DDS, "FastDDS Publisher released\n");
}

int FastDDSPublisher::Init() {
  InitParticipant();
  if (participant_ == nullptr)
    return DDSInitErrorType::DDSParticipantError;

  if (eprosima::fastrtps::xmlparser::XMLP_ret::XML_OK !=
      eprosima::fastrtps::xmlparser::XMLProfileManager::
      loadXMLFile(XML_FILE_PATH)) {
    return DDSInitErrorType::DDSTypeParserError;
  }
  eprosima::fastrtps::types::DynamicType_ptr dyn_type =
            eprosima::fastrtps::xmlparser::XMLProfileManager::
            getDynamicTypeByName("DDSMetaData")->build();
  dds_metadata_ = eprosima::fastrtps::types::DynamicDataFactory::
                  get_instance()->create_data(dyn_type);
  type_ = eprosima::fastrtps::types::DynamicPubSubType(dyn_type);
  type_.get()->auto_fill_type_information(false);
  type_.get()->auto_fill_type_object(true);
  type_.register_type(participant_);

  publisher_ = participant_->create_publisher(PUBLISHER_QOS_DEFAULT);
  if (publisher_ == nullptr)
    return DDSInitErrorType::DDSPublisherError;

  topic_ = participant_->create_topic(topic_name_,
                                      "DDSMetaData",
                                      TOPIC_QOS_DEFAULT);
  if (topic_ == nullptr)
    return DDSInitErrorType::DDSTopicError;

  listener_ = std::make_shared<FastDDSPubListener>();
  InitDataWriter();
  if (writer_ == nullptr)
    return DDSInitErrorType::DDSDataWriterError;

  LAVA_LOG(LOG_DDS, "Init Fast DDS Publisher Successfully, topic name: %s\n",
                    topic_name_.c_str());
  stop_ = false;
  return 0;
}

void FastDDSPublisher::InitDataWriter() {
  DataWriterQos wqos;
  wqos.history().kind = KEEP_LAST_HISTORY_QOS;
  wqos.history().depth = 30;
  wqos.resource_limits().max_samples = max_samples_;
  wqos.resource_limits().allocated_samples = max_samples_/2;
  wqos.reliable_writer_qos().times.heartbeatPeriod.seconds = 2;
  wqos.reliable_writer_qos().times.heartbeatPeriod.nanosec = 200 * 1000 * 1000;
  wqos.reliability().kind = RELIABLE_RELIABILITY_QOS;
  wqos.publish_mode().kind = ASYNCHRONOUS_PUBLISH_MODE;
  writer_ = publisher_->create_datawriter(topic_, wqos, listener_.get());
}

void FastDDSPublisher::InitParticipant() {
  DomainParticipantQos pqos;
  pqos.wire_protocol().builtin.discovery_config
                      .discoveryProtocol = DiscoveryProtocol_t::SIMPLE;
  pqos.wire_protocol().builtin.discovery_config
                      .use_SIMPLE_EndpointDiscoveryProtocol = true;
  pqos.wire_protocol().builtin.discovery_config.m_simpleEDP
                      .use_PublicationReaderANDSubscriptionWriter = true;
  pqos.wire_protocol().builtin.discovery_config.m_simpleEDP
                      .use_PublicationWriterANDSubscriptionReader = true;
  pqos.wire_protocol().builtin.discovery_config.leaseDuration =
                       eprosima::fastrtps::c_TimeInfinite;
  pqos.transport().use_builtin_transports = false;
  pqos.name("Participant pub" + topic_name_);

  auto transport_descriptor = GetTransportDescriptor(dds_transfer_type_);
  if (nullptr == transport_descriptor) {
    LAVA_LOG_ERR("Fatal: Create Transport Fault, exit\n");
    exit(-1);
  }
  pqos.transport().user_transports.push_back(transport_descriptor);
  
  if (dds_transfer_type_ == DDSTransportType::DDSTCPv4) {
    Locator_t initial_peer_locator;
    initial_peer_locator.kind = LOCATOR_KIND_TCPv4;
    IPLocator::setIPv4(initial_peer_locator, TCPv4_IP);
    initial_peer_locator.port = TCP_PORT;
    pqos.wire_protocol().builtin.initialPeersList.push_back(initial_peer_locator);
  }

  participant_ = DomainParticipantFactory::get_instance()
                                           ->create_participant(0, pqos);
}

bool FastDDSPublisher::Publish(MetaDataPtr metadata) {
  if (listener_->first_connected_ || listener_->matched_ > 0) {
    LAVA_DEBUG(LOG_DDS, "FastDDS publisher start publishing...\n");
    dds_metadata_->set_int64_value(metadata->nd, 0);
    dds_metadata_->set_int64_value(metadata->type, 1);
    dds_metadata_->set_int64_value(metadata->elsize, 2);
    dds_metadata_->set_int64_value(metadata->total_size, 3);
    LAVA_DEBUG(LOG_DDS, "FastDDS publisher set dims...\n");
    eprosima::fastrtps::types::DynamicData* array = dds_metadata_->loan_value(4);
    for (int i=0; i<5; i++)
      array->set_int64_value(metadata->dims[i], i);
    dds_metadata_->return_loaned_value(array);
    LAVA_DEBUG(LOG_DDS, "FastDDS publisher set strides...\n");
    array = dds_metadata_->loan_value(5);
    for (int i=0; i<5; i++)
      array->set_int64_value(metadata->strides[i], i);
    dds_metadata_->return_loaned_value(array);
    LAVA_DEBUG(LOG_DDS, "FastDDS publisher set mdata...\n");
    array = dds_metadata_->loan_value(6);
    char *ptr = (char*)metadata->mdata;
    for (int i=0; i<nbytes_; i++)
      array->set_char8_value(ptr[i], i);
    dds_metadata_->return_loaned_value(array);
    LAVA_DEBUG(LOG_DDS, "FastDDS publisher set data ok...\n");

    if (writer_->write(dds_metadata_.get()) != ReturnCode_t::RETCODE_OK) {
      LAVA_LOG_WARN(LOG_DDS, "Publisher write return not OK, Why work?\n");
    } else {
      LAVA_DEBUG(LOG_DDS, "Publish a data\n");
    }
    return true;
  }
  // LAVA_LOG_ERR("No listener matched\n");
  return false;
}

void FastDDSPublisher::Stop() {
  LAVA_LOG(LOG_DDS, "Stop FastDDS Publisher, waiting unmatched...\n");
  while (listener_->matched_ > 0) {
    helper::Sleep();
  }
  if (writer_ != nullptr){
    publisher_->delete_datawriter(writer_);
  }
  if (publisher_ != nullptr){
    participant_->delete_publisher(publisher_);
  }
  if (topic_ != nullptr){
    topic_->close();
    participant_->delete_topic(topic_);
  }
  if (participant_ != nullptr){
    DomainParticipantFactory::get_instance()->delete_participant(participant_);
  }
  stop_ = true;
}

void FastDDSPubListener::on_publication_matched(
        eprosima::fastdds::dds::DataWriter*,
        const eprosima::fastdds::dds::PublicationMatchedStatus& info) {
  if (info.current_count_change == 1) {
    matched_++;
    first_connected_ = true;
    LAVA_LOG(LOG_DDS, "FastDDS DataReader %d matched.\n", matched_);

  } else if (info.current_count_change == -1) {
    matched_--;
    LAVA_LOG(LOG_DDS, "FastDDS DataReader unmatched. matched_:%d\n", matched_);
  } else {
    LAVA_LOG_ERR("FastDDS Publistener status error\n");
  }
}

void FastDDSSubListener::on_subscription_matched(
        DataReader*,
        const SubscriptionMatchedStatus& info) {
  if (info.current_count_change == 1) {
    matched_++;
    LAVA_LOG(LOG_DDS, "FastDDS DataWriter %d matched.\n", matched_);
  } else if (info.current_count_change == -1) {
    matched_--;
    LAVA_LOG(LOG_DDS, "FastDDS DataWriter unmatched. matched_:%d\n", matched_);
  } else {
    LAVA_LOG_ERR("Subscriber number is not matched\n");
  }
}

FastDDSSubscriber::~FastDDSSubscriber() {
  LAVA_LOG(LOG_DDS, "FastDDS Subscriber Releasing...\n");
  if (!stop_) {
    LAVA_LOG_WARN(LOG_DDS, "Please stop Subscriber before release it next time\n");
    Stop();
  }
  LAVA_DEBUG(LOG_DDS, "FastDDS Subscriber Released...\n");
}

void FastDDSSubscriber::InitParticipant() {
  DomainParticipantQos pqos;
  pqos.wire_protocol().builtin.discovery_config.discoveryProtocol
                       = DiscoveryProtocol_t::SIMPLE;
  pqos.wire_protocol().builtin.discovery_config.
                       use_SIMPLE_EndpointDiscoveryProtocol = true;
  pqos.wire_protocol().builtin.discovery_config.m_simpleEDP.
                       use_PublicationReaderANDSubscriptionWriter = true;
  pqos.wire_protocol().builtin.discovery_config.m_simpleEDP.
                       use_PublicationWriterANDSubscriptionReader = true;
  pqos.wire_protocol().builtin.discovery_config.leaseDuration
                       = eprosima::fastrtps::c_TimeInfinite;
  pqos.transport().use_builtin_transports = false;
  pqos.name("Participant sub" + topic_name_);

  auto transport_descriptor = GetTransportDescriptor(dds_transfer_type_);
  if (nullptr == transport_descriptor) {
    LAVA_LOG_ERR("Fatal: Create Transport Fault, exit\n");
    exit(-1);
  }
  pqos.transport().user_transports.push_back(transport_descriptor);

  participant_ = DomainParticipantFactory::get_instance()
                                           ->create_participant(0, pqos);
}

void FastDDSSubscriber::InitDataReader() {
  DataReaderQos rqos;
  rqos.history().kind = KEEP_LAST_HISTORY_QOS;
  rqos.history().depth = 30;
  rqos.resource_limits().max_samples = max_samples_;
  rqos.resource_limits().allocated_samples = max_samples_/2;
  rqos.reliability().kind = RELIABLE_RELIABILITY_QOS;
  rqos.durability().kind = TRANSIENT_LOCAL_DURABILITY_QOS;

  reader_ = subscriber_->create_datareader(topic_, rqos, listener_.get());
}

int FastDDSSubscriber::Init() {
  InitParticipant();
  if (participant_ == nullptr)
    return DDSInitErrorType::DDSParticipantError;

  if (eprosima::fastrtps::xmlparser::XMLP_ret::XML_OK !=
      eprosima::fastrtps::xmlparser::XMLProfileManager::
      loadXMLFile(XML_FILE_PATH)) {
    return DDSInitErrorType::DDSTypeParserError;
  }
  eprosima::fastrtps::types::DynamicType_ptr dyn_type =
            eprosima::fastrtps::xmlparser::XMLProfileManager::
            getDynamicTypeByName("DDSMetaData")->build();
  dds_metadata_ = eprosima::fastrtps::types::DynamicDataFactory::
                  get_instance()->create_data(dyn_type);
  type_ = eprosima::fastrtps::types::DynamicPubSubType(dyn_type);

  type_.get()->auto_fill_type_information(false);
  type_.get()->auto_fill_type_object(true);
  type_.register_type(participant_);
  subscriber_ = participant_->create_subscriber(SUBSCRIBER_QOS_DEFAULT);
  if (subscriber_ == nullptr)
    return DDSInitErrorType::DDSSubscriberError;

  topic_ = participant_->create_topic(topic_name_,
                                      "DDSMetaData",
                                      TOPIC_QOS_DEFAULT);
  if (topic_ == nullptr)
    return DDSInitErrorType::DDSTopicError;

  listener_ = std::make_shared<FastDDSSubListener>();
  InitDataReader();
  if (reader_ == nullptr)
    return DDSInitErrorType::DDSDataReaderError;

  LAVA_LOG(LOG_DDS, "Init FastDDS Subscriber Successfully, topic name: %s\n",
                    topic_name_.c_str());
  stop_ = false;
  return 0;
}

MetaDataPtr FastDDSSubscriber::Read() {
  SampleInfo info;
  while (ReturnCode_t::RETCODE_OK != reader_
         ->take_next_sample(dds_metadata_.get(), &info)) {
    helper::Sleep();
  }

  if (info.valid_data) {
    // Recv data here
    LAVA_DEBUG(LOG_DDS, "FastDDS subscriber get metadata...\n");
    MetaDataPtr metadata = std::make_shared<MetaData>();
    dds_metadata_->get_int64_value(metadata->nd, 0);
    dds_metadata_->get_int64_value(metadata->type, 1);
    dds_metadata_->get_int64_value(metadata->elsize, 2);
    dds_metadata_->get_int64_value(metadata->total_size, 3);
    LAVA_DEBUG(LOG_DDS, "FastDDS subscriber get dims...\n");
    eprosima::fastrtps::types::DynamicData* array = dds_metadata_->loan_value(4);
    for (int i=0; i<5; i++)
      array->get_int64_value(metadata->dims[i], i);
    dds_metadata_->return_loaned_value(array);
    LAVA_DEBUG(LOG_DDS, "FastDDS subscriber get strides...\n");
    array = dds_metadata_->loan_value(5);
    for (int i=0; i<5; i++)
      array->get_int64_value(metadata->strides[i], i);
    dds_metadata_->return_loaned_value(array);
    LAVA_DEBUG(LOG_DDS, "FastDDS subscriber get mdata...\n");
    array = dds_metadata_->loan_value(6);
    char *ptr = (char*)malloc(nbytes_);
    for (int i=0; i<nbytes_; i++)
      array->get_char8_value(ptr[i], i);
    dds_metadata_->return_loaned_value(array);
    metadata->mdata = ptr;
    LAVA_DEBUG(LOG_DDS, "FastDDS subscriber get metadata ok...\n");

    LAVA_DEBUG(LOG_DDS, "Data Recieved, total_size:%d\n", metadata->total_size);
    return metadata;
  } else {
    LAVA_LOG_WARN(LOG_DDS, "Remote writer die\n");
  }

  LAVA_LOG_ERR("time out and no data received\n");
  return nullptr;
}

void FastDDSSubscriber::Stop() {
  LAVA_LOG(LOG_DDS, "Subscriber Stop and release\n");
  bool valid = true;
  if (reader_ != nullptr){
    subscriber_->delete_datareader(reader_);
  } else {
    valid = false;
  }
  if (topic_ != nullptr){
    participant_->delete_topic(topic_);
  } else {
    valid = false;
  }
  if (subscriber_ != nullptr) {
    participant_->delete_subscriber(subscriber_);
  } else {
    valid = false;
  }
  if (participant_ != nullptr){
    DomainParticipantFactory::get_instance()->delete_participant(participant_);
  } else {
    valid = false;
  }
  if (!valid) {
    LAVA_LOG_ERR("Stop function is not valid\n");
  }
  stop_ = true;
}

std::shared_ptr<eprosima::fastdds::rtps::TransportDescriptorInterface>
GetTransportDescriptor(const DDSTransportType &dds_type) {
  if (dds_type == DDSTransportType::DDSSHM) {
    LAVA_LOG(LOG_DDS, "Shared Memory Transport Descriptor\n");
    auto transport = std::make_shared<SharedMemTransportDescriptor>();
    transport->segment_size(SHM_SEGMENT_SIZE);
    return transport;
  } else if (dds_type == DDSTransportType::DDSTCPv4) {
    LAVA_LOG(LOG_DDS, "TCPv4 Transport Descriptor\n");
    auto transport = std::make_shared<TCPv4TransportDescriptor>();
    transport->set_WAN_address(TCPv4_IP);
    transport->add_listener_port(TCP_PORT);
    transport->interfaceWhiteList.push_back(TCPv4_IP); // loopback
    return transport;
  } else if (dds_type == DDSTransportType::DDSUDPv4) {
    LAVA_LOG(LOG_DDS, "UDPv4 Transport Descriptor\n");
    auto transport = std::make_shared<UDPv4TransportDescriptor>();
    transport->m_output_udp_socket = UDP_OUT_PORT;
    transport->non_blocking_send = NON_BLOCKING_SEND;
    return transport;
  } else {
    LAVA_LOG_ERR("TransportType %d has not supported\n", dds_type);
  }
  return nullptr;
}

}  // namespace message_infrastructure
