// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef CHANNEL_DDS_FAST_DDS_H_
#define CHANNEL_DDS_FAST_DDS_H_

#include <channel/dds/dds.h>
#include <channel/dds/protos/fast_dds/DDSMetaDataPubSubTypes.h>
#include <channel/dds/protos/fast_dds/DDSMetaData.h>
#include <core/utils.h>
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

namespace message_infrastructure {

class FastDDSPubListener final : public
                                 eprosima::fastdds::dds::DataWriterListener {
 public:
  FastDDSPubListener() : matched_(0) {}
  ~FastDDSPubListener() override {}
  void on_publication_matched(
    eprosima::fastdds::dds::DataWriter* writer,
    const eprosima::fastdds::dds::PublicationMatchedStatus& info) override;

  int matched_;
};

// FastDDSPubListener object needs to be transfered to DDSPort.
// Also need to be handled in DDS class.
// Use std::shared_ptr.
using FastDDSPubListenerPtr = std::shared_ptr<FastDDSPubListener>;

class FastDDSPublisher final : public DDSPublisher {
 public:
  FastDDSPublisher(const std::string &topic_name,
                   const DDSTransportType &dds_transfer_type,
                   const size_t &max_samples) :
                   type_(new ddsmetadata::msg::DDSMetaDataPubSubType()),
                   stop_(true),
                   topic_name_(topic_name),
                   dds_transfer_type_(dds_transfer_type),
                   max_samples_(max_samples) {}
  ~FastDDSPublisher() override;
  DDSInitErrorType Init();
  bool Publish(DataPtr data);
  void Stop();  // Can Init again

 private:
  void InitDataWriter();
  void InitParticipant();

  FastDDSPubListenerPtr listener_ = nullptr;
  std::shared_ptr<ddsmetadata::msg::DDSMetaData> dds_metadata_;
  eprosima::fastdds::dds::DomainParticipant* participant_ = nullptr;
  eprosima::fastdds::dds::Publisher* publisher_ = nullptr;
  eprosima::fastdds::dds::Topic* topic_ = nullptr;
  eprosima::fastdds::dds::DataWriter* writer_ = nullptr;
  eprosima::fastdds::dds::TypeSupport type_;

  std::string topic_name_;
  DDSTransportType dds_transfer_type_;
  size_t max_samples_;

  bool stop_;
};

class FastDDSSubListener final : public
                         eprosima::fastdds::dds::DataReaderListener {
 public:
  FastDDSSubListener() : matched_(0) {}
  ~FastDDSSubListener() override {}
  void on_data_available(
       eprosima::fastdds::dds::DataReader* reader) override {};
  void on_subscription_matched(
       eprosima::fastdds::dds::DataReader* reader,
       const eprosima::fastdds::dds::SubscriptionMatchedStatus& info) override;
  int matched_;
};

// FastDDSSubListener object needs to be transfered to DDSPort.
// Also need to be handled in DDS class.
// Use std::shared_ptr.
using FastDDSSubListenerPtr = std::shared_ptr<FastDDSSubListener>;

class FastDDSSubscriber final : public DDSSubscriber {
 public:
  FastDDSSubscriber(const std::string &topic_name,
                    const DDSTransportType &dds_transfer_type,
                    const size_t &max_samples) :
                    type_(new ddsmetadata::msg::DDSMetaDataPubSubType()),
                    stop_(true),
                    topic_name_(topic_name),
                    dds_transfer_type_(dds_transfer_type),
                    max_samples_(max_samples) {}
  ~FastDDSSubscriber() override;
  DDSInitErrorType Init();
  void Stop();
  MetaDataPtr Recv(bool keep);
  bool Probe();

 private:
  void InitParticipant();
  void InitDataReader();
  FastDDSSubListenerPtr listener_ = nullptr;
  eprosima::fastdds::dds::DomainParticipant* participant_ = nullptr;
  eprosima::fastdds::dds::Subscriber* subscriber_ = nullptr;
  eprosima::fastdds::dds::Topic* topic_ = nullptr;
  eprosima::fastdds::dds::DataReader* reader_ = nullptr;
  eprosima::fastdds::dds::TypeSupport type_;

  std::string topic_name_;
  DDSTransportType dds_transfer_type_;
  size_t max_samples_;
  bool stop_;
};

std::shared_ptr<eprosima::fastdds::rtps::TransportDescriptorInterface>
GetTransportDescriptor(const DDSTransportType &dds_type);
}  // namespace message_infrastructure

#endif  // CHANNEL_DDS_FAST_DDS_H_
