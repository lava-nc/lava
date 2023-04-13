// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/


#include <channel/dds/fast_dds.h>
#include <core/message_infrastructure_logging.h>

#include <fastdds/rtps/transport/shared_mem/SharedMemTransportDescriptor.h>
#include <fastdds/rtps/transport/TCPv4TransportDescriptor.h>
#include <fastdds/rtps/transport/TCPv6TransportDescriptor.h>
#include <fastdds/rtps/transport/UDPv4TransportDescriptor.h>
#include <fastdds/rtps/transport/UDPv6TransportDescriptor.h>
#include <fastrtps/Domain.h>

#include <vector>
#include <fastdds/dds/domain/DomainParticipantFactory.hpp>

namespace message_infrastructure {

using namespace eprosima::fastdds::dds;  // NOLINT
using namespace eprosima::fastdds::rtps;  // NOLINT
using namespace eprosima::fastrtps::rtps;  // NOLINT

FastDDSPublisher::~FastDDSPublisher() {
  LAVA_DEBUG(LOG_DDS, "FastDDS Publisher releasing...\n");
  if (!stop_) {
    LAVA_LOG_WARN(LOG_DDS, "Code should Stop Publisher\n");
    Stop();
  }
  LAVA_DEBUG(LOG_DDS, "FastDDS Publisher released\n");
}

DDSInitErrorType FastDDSPublisher::Init() {

  dds_metadata_ = std::make_shared<ddsmetadata::msg::DDSMetaData>();
  InitParticipant();
  if (participant_ == nullptr)
    return DDSInitErrorType::DDSParticipantError;
    
  type_.register_type(participant_);
  publisher_ = participant_->create_publisher(PUBLISHER_QOS_DEFAULT);
  if (publisher_ == nullptr)
    return DDSInitErrorType::DDSPublisherError;

  topic_ = participant_->create_topic(topic_name_,
                                      DDS_DATATYPE_NAME,
                                      TOPIC_QOS_DEFAULT);
  if (topic_ == nullptr)
    return DDSInitErrorType::DDSTopicError;

  listener_ = std::make_shared<FastDDSPubListener>();
  InitDataWriter();
  if (writer_ == nullptr)
    return DDSInitErrorType::DDSDataWriterError;

  LAVA_DEBUG(LOG_DDS, "Init Fast DDS Publisher Successfully, topic name: %s\n",
                    topic_name_.c_str());
  stop_ = false;
  return DDSInitErrorType::DDSNOERR;
}

void FastDDSPublisher::InitDataWriter() {
  DataWriterQos wqos;
  wqos.history().kind = KEEP_ALL_HISTORY_QOS;
  wqos.history().depth = max_samples_;
  wqos.resource_limits().max_samples = max_samples_;
  wqos.resource_limits().allocated_samples = max_samples_ / 2;
  wqos.reliable_writer_qos().times
                            .heartbeatPeriod.seconds = HEARTBEAT_PERIOD_SECONDS;
  wqos.reliable_writer_qos().times
                            .heartbeatPeriod.nanosec = HEARTBEAT_PERIOD_NANOSEC;
  wqos.reliability().kind = RELIABLE_RELIABILITY_QOS;
  wqos.publish_mode().kind = ASYNCHRONOUS_PUBLISH_MODE;
  wqos.endpoint().history_memory_policy = PREALLOCATED_WITH_REALLOC_MEMORY_MODE;
  writer_ = publisher_->create_datawriter(topic_, wqos, listener_.get());
}

void FastDDSPublisher::InitParticipant() {
  DomainParticipantQos pqos;
  pqos.transport().use_builtin_transports = false;
  pqos.name("Participant pub" + topic_name_);

  auto transport_descriptor = GetTransportDescriptor(dds_transfer_type_);
  if (nullptr == transport_descriptor) {
    LAVA_LOG_FATAL("Create Transport Fault, exit\n");
  }
  pqos.transport().user_transports.push_back(transport_descriptor);

  if (dds_transfer_type_ == DDSTransportType::DDSTCPv4) {
    Locator_t initial_peer_locator;
    initial_peer_locator.kind = LOCATOR_KIND_TCPv4;
    IPLocator::setIPv4(initial_peer_locator, TCPv4_IP);
    initial_peer_locator.port = TCP_PORT;
    pqos.wire_protocol().builtin.initialPeersList
                        .push_back(initial_peer_locator);
  }

  participant_ = DomainParticipantFactory::get_instance()
                                           ->create_participant(0, pqos);
}

bool FastDDSPublisher::Publish(DataPtr data) {
  LAVA_LOG_ERR("FastDDSPublisher::Publish topic name = %s\n", topic_name_.c_str());
  MetaData* metadata = reinterpret_cast<MetaData*>(data.get());
  if (listener_->matched_ > 0) {
    LAVA_DEBUG(LOG_DDS, "FastDDS publisher start publishing...\n");
    dds_metadata_->nd(metadata->nd);
    dds_metadata_->type(metadata->type);
    dds_metadata_->elsize(metadata->elsize);
    dds_metadata_->total_size(metadata->total_size);
    memcpy(&dds_metadata_->dims()[0], metadata->dims, sizeof(metadata->dims));
    memcpy(&dds_metadata_->strides()[0],
           metadata->strides,
           sizeof(metadata->strides));
    size_t nbytes = metadata->elsize * metadata->total_size;
    dds_metadata_->mdata(std::vector<unsigned char>(
                   reinterpret_cast<unsigned char*>(metadata->mdata),
                   reinterpret_cast<unsigned char*>(metadata->mdata) + nbytes));
    LAVA_DEBUG(LOG_DDS, "FastDDS publisher copied\n");

    if (writer_->write(dds_metadata_.get()) != ReturnCode_t::RETCODE_OK) {
      LAVA_LOG_WARN(LOG_DDS, "Publisher write return not OK, Why work?\n");
    } else {
      LAVA_DEBUG(LOG_DDS, "Publish a data\n");
    }
    return true;
  }
  return false;
}

void FastDDSPublisher::Stop() {
  LAVA_LOG(LOG_DDS, "Stop FastDDS Publisher, waiting unmatched...\n");
  while (listener_ != nullptr && listener_->matched_ > 0) {
    helper::Sleep();
  }
  if (writer_ != nullptr) {
    LAVA_LOG_ERR("pub delete_datawriter\n");
    publisher_->delete_datawriter(writer_);
  }
  if (publisher_ != nullptr) {
    LAVA_LOG_ERR("pub delete_publisher\n");
    participant_->delete_publisher(publisher_);
  }
  if (topic_ != nullptr) {
    LAVA_LOG_ERR("pub delete_topic\n");
    topic_->close();
    participant_->delete_topic(topic_);
  }
  if (participant_ != nullptr) {
    DomainParticipantFactory::get_instance()->delete_participant(participant_);
  }
  stop_ = true;
}

void FastDDSPubListener::on_publication_matched(
        eprosima::fastdds::dds::DataWriter*,
        const eprosima::fastdds::dds::PublicationMatchedStatus& info) {
  if (info.current_count_change == 1) {
    matched_++;
    LAVA_DEBUG(LOG_DDS, "FastDDS DataReader %d matched.\n", matched_);
  } else if (info.current_count_change == -1) {
    matched_--;
    LAVA_DEBUG(LOG_DDS, "FastDDS DataReader unmatched, remain:%d\n", matched_);
  } else {
    LAVA_LOG_ERR("FastDDS Publistener status error\n");
  }
}

void FastDDSSubListener::on_subscription_matched(
        DataReader*,
        const SubscriptionMatchedStatus& info) {
  if (info.current_count_change == 1) {
    matched_++;
    LAVA_DEBUG(LOG_DDS, "FastDDS DataWriter %d matched.\n", matched_);
  } else if (info.current_count_change == -1) {
    matched_--;
    LAVA_DEBUG(LOG_DDS, "FastDDS DataWriter unmatched, remain:%d\n", matched_);
  } else {
    LAVA_LOG_ERR("Subscriber number is not matched\n");
  }
}

FastDDSSubscriber::~FastDDSSubscriber() {
  LAVA_DEBUG(LOG_DDS, "FastDDS Subscriber Releasing...\n");
  if (!stop_) {
    LAVA_LOG_WARN(LOG_DDS, "Code should Stop Subscriber\n");
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
    LAVA_LOG_FATAL("Create Transport Fault, exit\n");
  }
  pqos.transport().user_transports.push_back(transport_descriptor);

  participant_ = DomainParticipantFactory::get_instance()
                                           ->create_participant(0, pqos);
}

void FastDDSSubscriber::InitDataReader() {
  DataReaderQos rqos;
  rqos.history().kind = KEEP_ALL_HISTORY_QOS;
  rqos.history().depth = max_samples_;
  rqos.resource_limits().max_samples = max_samples_;
  rqos.resource_limits().allocated_samples = max_samples_ / 2;
  rqos.reliability().kind = RELIABLE_RELIABILITY_QOS;
  rqos.durability().kind = TRANSIENT_LOCAL_DURABILITY_QOS;
  rqos.endpoint().history_memory_policy = PREALLOCATED_WITH_REALLOC_MEMORY_MODE;
  reader_ = subscriber_->create_datareader(topic_, rqos, listener_.get());
}

DDSInitErrorType FastDDSSubscriber::Init() {
  InitParticipant();
  if (participant_ == nullptr)
    return DDSInitErrorType::DDSParticipantError;

  type_.register_type(participant_);
  subscriber_ = participant_->create_subscriber(SUBSCRIBER_QOS_DEFAULT);
  if (subscriber_ == nullptr)
    return DDSInitErrorType::DDSSubscriberError;

  topic_ = participant_->create_topic(topic_name_,
                                      DDS_DATATYPE_NAME,
                                      TOPIC_QOS_DEFAULT);
  if (topic_ == nullptr)
    return DDSInitErrorType::DDSTopicError;

  listener_ = std::make_shared<FastDDSSubListener>();
  InitDataReader();
  if (reader_ == nullptr)
    return DDSInitErrorType::DDSDataReaderError;

  LAVA_DEBUG(LOG_DDS, "Init FastDDS Subscriber Successfully, topic name: %s\n",
                      topic_name_.c_str());
  stop_ = false;
  return DDSInitErrorType::DDSNOERR;
}

MetaDataPtr FastDDSSubscriber::Recv(bool keep) {
  LAVA_LOG_ERR("FastDDSSubscriber::Recv topic name = %s\n", topic_name_.c_str());
  FASTDDS_CONST_SEQUENCE(MDataSeq, ddsmetadata::msg::DDSMetaData);
  MDataSeq mdata_seq;
  SampleInfoSeq infos;
  if (keep) {
    LAVA_DEBUG(LOG_DDS, "Keep the data recieved\n");
    while (ReturnCode_t::RETCODE_OK !=
           reader_->read(mdata_seq, infos, 1)) {
      helper::Sleep();
    }
  } else {
    LAVA_DEBUG(LOG_DDS, "Take the data recieved\n");

    while (ReturnCode_t::RETCODE_OK !=
           reader_->take(mdata_seq, infos, 1)) {
      helper::Sleep();
    }
    LAVA_DEBUG(LOG_DDS, "Taked the data recieved==\n");

  }

  LAVA_DEBUG(LOG_DDS, "Return the data recieved\n");
  LAVA_DEBUG(LOG_DDS, "INFO length: %d\n", infos.length());
  if (infos[0].valid_data) {
    const ddsmetadata::msg::DDSMetaData& dds_metadata = mdata_seq[0];
    MetaDataPtr metadata = std::make_shared<MetaData>();
    metadata->nd = dds_metadata.nd();
    metadata->type = dds_metadata.type();
    metadata->elsize = dds_metadata.elsize();
    metadata->total_size = dds_metadata.total_size();
    memcpy(metadata->dims, dds_metadata.dims().data(), sizeof(metadata->dims));
    memcpy(metadata->strides,
           dds_metadata.strides().data(),
           sizeof(metadata->strides));
    int nbytes = metadata->elsize * metadata->total_size;
    void *ptr = std::calloc(nbytes, 1);
    if (ptr == nullptr) {
      LAVA_LOG_ERR("alloc failed, errno: %d\n", errno);
    }
    memcpy(ptr, dds_metadata.mdata().data(), nbytes);
    metadata->mdata = ptr;
    reader_->return_loan(mdata_seq, infos);
    LAVA_DEBUG(LOG_DDS, "Data Recieved\n");
    return metadata;
  } else {
    LAVA_LOG_WARN(LOG_DDS, "Remote writer die\n");
  }

  LAVA_LOG_ERR("time out and no data received\n");
  return nullptr;
}

bool FastDDSSubscriber::Probe() {
  FASTDDS_CONST_SEQUENCE(MDataSeq, ddsmetadata::msg::DDSMetaData);
  MDataSeq mdata_seq;
  SampleInfoSeq infos;
  bool res = false;
  if (ReturnCode_t::RETCODE_OK == reader_->read(mdata_seq, infos, 1)) {
    reader_->return_loan(mdata_seq, infos);
    res = true;
  }
  return res;
}

void FastDDSSubscriber::Stop() {
  LAVA_DEBUG(LOG_DDS, "Subscriber Stop and release\n");
  bool valid = true;
  if (reader_ != nullptr) {
    LAVA_LOG_ERR("sub delete_topic\n");
    subscriber_->delete_datareader(reader_);
  } else {
    valid = false;
  }
  if (topic_ != nullptr) {
    LAVA_LOG_ERR("sub delete_topic\n");
    participant_->delete_topic(topic_);
  } else {
    LAVA_LOG_ERR("topic_ == nullptr\n");
    valid = false;
  }
  if (subscriber_ != nullptr) {
    LAVA_LOG_ERR("sub delete_topic\n");
    participant_->delete_subscriber(subscriber_);
  } else {
    valid = false;
  }
  if (participant_ != nullptr) {
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
    LAVA_DEBUG(LOG_DDS, "Shared Memory Transport Descriptor\n");
    auto transport = std::make_shared<SharedMemTransportDescriptor>();
    transport->segment_size(SHM_SEGMENT_SIZE);
    return transport;
  } else if (dds_type == DDSTransportType::DDSTCPv4) {
    LAVA_DEBUG(LOG_DDS, "TCPv4 Transport Descriptor\n");
    auto transport = std::make_shared<TCPv4TransportDescriptor>();
    transport->set_WAN_address(TCPv4_IP);
    transport->add_listener_port(TCP_PORT);
    transport->interfaceWhiteList.push_back(TCPv4_IP);  // loopback
    return transport;
  } else if (dds_type == DDSTransportType::DDSUDPv4) {
    LAVA_DEBUG(LOG_DDS, "UDPv4 Transport Descriptor\n");
    auto transport = std::make_shared<UDPv4TransportDescriptor>();
    transport->m_output_udp_socket = UDP_OUT_PORT;
    transport->non_blocking_send = NON_BLOCKING_SEND;
    return transport;
  } else {
    LAVA_LOG_ERR("TransportType %d has not supported\n",
                 static_cast<int>(dds_type));
  }
  return nullptr;
}

}  // namespace message_infrastructure
