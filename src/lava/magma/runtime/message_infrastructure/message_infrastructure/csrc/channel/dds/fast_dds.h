// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef CHANNEL_DDS_FAST_DDS_H_
#define CHANNEL_DDS_FAST_DDS_H_

#include <message_infrastructure/csrc/channel/dds/dds.h>
#include <message_infrastructure/csrc/channel/dds/protos/fast_dds/metadataPubSubTypes.h>
#include <message_infrastructure/csrc/channel/dds/protos/fast_dds/metadata.h>
#include <message_infrastructure/csrc/core/utils.h>
#include <fastdds/rtps/transport/TransportDescriptorInterface.h>

#include <memory>
#include <string>

#include <fastdds/dds/domain/DomainParticipant.hpp>
#include <fastdds/dds/publisher/DataWriter.hpp>
#include <fastdds/dds/publisher/DataWriterListener.hpp>
#include <fastdds/dds/publisher/Publisher.hpp>
#include <fastdds/dds/subscriber/DataReader.hpp>
#include <fastdds/dds/subscriber/SampleInfo.hpp>
#include <fastdds/dds/subscriber/Subscriber.hpp>
#include <fastdds/dds/topic/Topic.hpp>
#include <fastdds/dds/topic/TypeSupport.hpp>

// Default Parameters
// Transport
#define SHM_SEGMENT_SIZE (2*1024*1024)
#define NON_BLOCKING_SEND (false)
#define UDP_OUT_PORT  (0)
#define TCP_PORT 46
#define TCPv4_IP ("0.0.0.0")
// QOS

namespace message_infrastructure {

class FastDDSPubListener final : public
                                 eprosima::fastdds::dds::DataWriterListener {
 public:
  FastDDSPubListener() : matched_(false) {}
  ~FastDDSPubListener() override {}
  void on_publication_matched(
    eprosima::fastdds::dds::DataWriter* writer,
    const eprosima::fastdds::dds::PublicationMatchedStatus& info) override;

  int matched_;
};

using FastDDSPubListenerPtr = std::shared_ptr<FastDDSPubListener>;

class FastDDSPublisher final : public DDSPublisher {
 public:
  FastDDSPublisher(const size_t &max_samples,
                   const size_t &nbytes,
                   const std::string &topic_name,
                   const DDSTransportType &dds_transfer_type) :
                   type_(new DDSMetaDataPubSubType()),
                   stop_(true),
                   max_samples_(max_samples),
                   nbytes_(nbytes),
                   topic_name_(topic_name),
                   dds_transfer_type_(dds_transfer_type) {}
  ~FastDDSPublisher() override;
  int Init();
  bool Publish(MetaDataPtr metadata);
  void Stop();  // Can Init again

 private:
  void InitDataWriter();
  void InitParticipant();

  FastDDSPubListenerPtr listener_ = nullptr;
  std::shared_ptr<DDSMetaData> dds_metadata_;
  eprosima::fastdds::dds::DomainParticipant* participant_ = nullptr;
  eprosima::fastdds::dds::Publisher* publisher_ = nullptr;
  eprosima::fastdds::dds::Topic* topic_ = nullptr;
  eprosima::fastdds::dds::DataWriter* writer_ = nullptr;
  eprosima::fastdds::dds::TypeSupport type_;

  size_t max_samples_;
  size_t nbytes_;
  std::string topic_name_;
  DDSTransportType dds_transfer_type_;

  bool stop_;
};

class FastDDSSubListener final : public
                         eprosima::fastdds::dds::DataReaderListener {
 public:
  FastDDSSubListener() : matched_(0), samples_(0) {}
  ~FastDDSSubListener() override {}
  void on_data_available(
       eprosima::fastdds::dds::DataReader* reader) override {};
  void on_subscription_matched(
       eprosima::fastdds::dds::DataReader* reader,
       const eprosima::fastdds::dds::SubscriptionMatchedStatus& info) override;
  int matched_;
  uint32_t samples_;
};

using FastDDSSubListenerPtr = std::shared_ptr<FastDDSSubListener>;

class FastDDSSubscriber final : public DDSSubscriber {
 public:
  FastDDSSubscriber(const size_t &max_samples,
                   const size_t &nbytes,
                   const std::string &topic_name,
                   const DDSTransportType &dds_transfer_type) :
                   type_(new DDSMetaDataPubSubType()),
                   stop_(true),
                   max_samples_(max_samples),
                   nbytes_(nbytes),
                   topic_name_(topic_name),
                   dds_transfer_type_(dds_transfer_type)
                    {};
  ~FastDDSSubscriber() override;
  int Init();
  void Stop();
  MetaDataPtr Recv(bool keep);

 private:
  void InitParticipant();
  void InitDataReader();
  FastDDSSubListenerPtr listener_ = nullptr;
  std::shared_ptr<DDSMetaData> dds_metadata_;
  eprosima::fastdds::dds::DomainParticipant* participant_ = nullptr;
  eprosima::fastdds::dds::Subscriber* subscriber_ = nullptr;
  eprosima::fastdds::dds::Topic* topic_ = nullptr;
  eprosima::fastdds::dds::DataReader* reader_ = nullptr;
  eprosima::fastdds::dds::TypeSupport type_;
  DDSTransportType dds_transfer_type_;
  int max_samples_;
  size_t nbytes_;
  std::string topic_name_;
  bool stop_;
};

std::shared_ptr<eprosima::fastdds::rtps::TransportDescriptorInterface>
GetTransportDescriptor(const DDSTransportType &dds_type);
}  // namespace message_infrastructure

#endif  // CHANNEL_DDS_FAST_DDS_H_
