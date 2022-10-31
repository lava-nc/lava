// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/


#include <message_infrastructure/csrc/channel/dds/fast_dds.h>
#include <message_infrastructure/csrc/core/message_infrastructure_logging.h>

#include <fastdds/dds/domain/DomainParticipantFactory.hpp>
#include <fastdds/rtps/transport/shared_mem/SharedMemTransportDescriptor.h>

namespace message_infrastructure {

using namespace eprosima::fastdds::dds;
using namespace eprosima::fastdds::rtps;
using namespace eprosima::fastrtps::rtps;

FastDDSPublisher::~FastDDSPublisher() {
  publisher_->delete_datawriter(writer_);
  participant_->delete_publisher(publisher_);
  participant_->delete_topic(topic_);
  DomainParticipantFactory::get_instance()->delete_participant(participant_);
}

int FastDDSPublisher::Init() {
  dds_metadata_ = std::make_shared<DDSMetaData>();
  InitParticipant();
  if (participant_ == nullptr)
    return -1;

  type_.register_type(participant_);
  publisher_ = participant_->create_publisher(PUBLISHER_QOS_DEFAULT);
  if (publisher_ == nullptr)
    return -2;

  topic_ = participant_->create_topic(topic_name_, "DDSMetaData", TOPIC_QOS_DEFAULT);
  if (topic_ == nullptr)
    return -3;

  listener_ = std::make_shared<FastDDSPubListener>();
  InitDataWriter();
  if (writer_ == nullptr)
    return -4;
  
  printf("Init Publisher Successfully, topic name: %s\n", topic_name_.c_str());
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
  pqos.wire_protocol().builtin.discovery_config.discoveryProtocol = DiscoveryProtocol_t::SIMPLE;
  pqos.wire_protocol().builtin.discovery_config.use_SIMPLE_EndpointDiscoveryProtocol = true;
  pqos.wire_protocol().builtin.discovery_config.m_simpleEDP.use_PublicationReaderANDSubscriptionWriter = true;
  pqos.wire_protocol().builtin.discovery_config.m_simpleEDP.use_PublicationWriterANDSubscriptionReader = true;
  pqos.wire_protocol().builtin.discovery_config.leaseDuration = eprosima::fastrtps::c_TimeInfinite;
  pqos.transport().use_builtin_transports = false;
  pqos.name("Participant pub" + topic_name_);
  
  auto shm_transport = std::make_shared<SharedMemTransportDescriptor>();
  shm_transport->segment_size(2 * 1024 * 1024);
  pqos.transport().user_transports.push_back(shm_transport);

  participant_ = DomainParticipantFactory::get_instance()->create_participant(0, pqos);
}

bool FastDDSPublisher::Publish(MetaDataPtr metadata) {
  if (listener_->first_connected_ || listener_->matched_ > 0) {
    printf("start publishing...\n");
    memcpy(&dds_metadata_->mdata()[0], metadata.get(), sizeof(MetaData));
    memcpy(&dds_metadata_->mdata()[sizeof(MetaData)], metadata->mdata,
            metadata->elsize * metadata->total_size);
    printf("medata copied %d mdata\n", metadata->elsize * metadata->total_size);
    if(writer_->write(dds_metadata_.get()) != ReturnCode_t::RETCODE_OK) {
      printf("what error?\n");
    }
    else {
      printf("Publish a data\n");
    }
    return true;
  }
  // LAVA_LOG_ERR("No listener matched\n");
  return false;
}

void FastDDSPublisher::Stop() {
  printf("stop publisher and do something\n");
  while(listener_->matched_) {
    helper::Sleep();
  }
}

void FastDDSPubListener::on_publication_matched(
        eprosima::fastdds::dds::DataWriter*,
        const eprosima::fastdds::dds::PublicationMatchedStatus& info) {
  if (info.current_count_change == 1) {
    matched_ = true;
    first_connected_ = true;
    printf("Publisher matched.\n");
  }
  else if (info.current_count_change == -1) {
    matched_ = false;
    printf("Publisher unmatched. matched_:%d\n", matched_);
  }
  else
  {
    LAVA_LOG_ERR(" is not a valid value for PublicationMatchedStatus current count change\n");
  }
}

void FastDDSSubListener::on_subscription_matched(
        DataReader*,
        const SubscriptionMatchedStatus& info) {
  if (info.current_count_change == 1) {
    matched_ = info.total_count;
    printf("Subscriber matched.\n");
  }
  else if (info.current_count_change == -1) {
    matched_ = info.total_count;
    printf("Subscriber unmatched. matched_:%d\n", matched_);
  }
  else {
    LAVA_LOG_ERR("Subscriber number is not matched\n");
  }
}

FastDDSSubscriber::~FastDDSSubscriber() {
  subscriber_->delete_datareader(reader_);
  participant_->delete_topic(topic_);
  participant_->delete_subscriber(subscriber_);
  DomainParticipantFactory::get_instance()->delete_participant(participant_);
}

void FastDDSSubscriber::InitParticipant() {
  DomainParticipantQos pqos;
  pqos.wire_protocol().builtin.discovery_config.discoveryProtocol = DiscoveryProtocol_t::SIMPLE;
  pqos.wire_protocol().builtin.discovery_config.use_SIMPLE_EndpointDiscoveryProtocol = true;
  pqos.wire_protocol().builtin.discovery_config.m_simpleEDP.use_PublicationReaderANDSubscriptionWriter = true;
  pqos.wire_protocol().builtin.discovery_config.m_simpleEDP.use_PublicationWriterANDSubscriptionReader = true;
  pqos.wire_protocol().builtin.discovery_config.leaseDuration = eprosima::fastrtps::c_TimeInfinite;
  pqos.transport().use_builtin_transports = false;
  pqos.name("Participant sub" + topic_name_);
  
  auto shm_transport = std::make_shared<SharedMemTransportDescriptor>();
  shm_transport->segment_size(2 * 1024 * 1024); // TODO: size
  pqos.transport().user_transports.push_back(shm_transport);

  participant_ = DomainParticipantFactory::get_instance()->create_participant(0, pqos);
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
  dds_metadata_ = std::make_shared<DDSMetaData>();
  InitParticipant();
  if (participant_ == nullptr)
    return -1;

  type_.register_type(participant_);
  subscriber_ = participant_->create_subscriber(SUBSCRIBER_QOS_DEFAULT);
  if (subscriber_ == nullptr)
    return -2;

  topic_ = participant_->create_topic(topic_name_, "DDSMetaData", TOPIC_QOS_DEFAULT);
  if (topic_ == nullptr)
    return -3;

  listener_ = std::make_shared<FastDDSSubListener>();
  InitDataReader();
  if (reader_ == nullptr)
    return -4;

  printf("Init Subscriber Successfully, topic name: %s\n", topic_name_.c_str());
  return 0;
}

MetaDataPtr FastDDSSubscriber::Read() {
  SampleInfo info;
  while (ReturnCode_t::RETCODE_OK != reader_->take_next_sample(dds_metadata_.get(), &info)) {
    helper::Sleep();
  }

  if (info.valid_data) {
    // Recv data here
    MetaDataPtr metadata = std::make_shared<MetaData>();
    memcpy(metadata.get(), dds_metadata_->mdata().data(), sizeof(MetaData));
    printf("Allocating %d size\n", metadata->elsize * metadata->total_size);
    void *ptr = malloc(metadata->elsize * metadata->total_size);
    memcpy(ptr, dds_metadata_->mdata().data()+sizeof(MetaData),
                                    metadata->elsize * metadata->total_size);
    metadata->mdata = ptr;
    printf("Data Recieved\n");
    return metadata;
  }
  else {
      printf("Remote writer die\n");
  }

  LAVA_LOG_ERR("time out and no data received\n");
  return nullptr;

}

}  // namespace message_infrastructure