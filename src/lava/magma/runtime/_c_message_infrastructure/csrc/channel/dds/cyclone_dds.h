// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef CHANNEL_DDS_CYCLONE_DDS_H_
#define CHANNEL_DDS_CYCLONE_DDS_H_

#include <channel/dds/dds.h>
#include <core/utils.h>
#include <atomic>
#include <memory>
#include <string>
#include <dds/dds.hpp>
#include <channel/dds/protos/cyclone_dds/DDSMetaData.hpp>

namespace message_infrastructure {

class CycloneDDSPubListener final : public
      dds::pub::NoOpDataWriterListener<ddsmetadata::msg::DDSMetaData>{
 public:
  CycloneDDSPubListener() : matched_(0) {}
  void on_offered_incompatible_qos(
    dds::pub::DataWriter<ddsmetadata::msg::DDSMetaData>& writer,
    const dds::core::status::OfferedIncompatibleQosStatus&  status) override;
  void on_publication_matched(
    dds::pub::DataWriter<ddsmetadata::msg::DDSMetaData> &writer,
    const dds::core::status::PublicationMatchedStatus &info) override;
  ~CycloneDDSPubListener() override {}
  std::atomic_uint32_t matched_;
};

using CycloneDDSPubListenerPtr = std::shared_ptr<CycloneDDSPubListener>;

class CycloneDDSPublisher final : public DDSPublisher {
 public:
  CycloneDDSPublisher(const std::string &topic_name,
                      const DDSTransportType &dds_transfer_type,
                      const size_t &max_sample) :
                      stop_(true),
                      topic_name_(topic_name),
                      dds_transfer_type_(dds_transfer_type),
                      max_samples_(max_sample) {}
  ~CycloneDDSPublisher() override;
  DDSInitErrorType Init();
  bool Publish(DataPtr metadata);
  void Stop();  // Can Init again

 private:
  CycloneDDSPubListenerPtr listener_ = nullptr;
  std::shared_ptr<ddsmetadata::msg::DDSMetaData> dds_metadata_ = nullptr;
  dds::domain::DomainParticipant participant_ = dds::core::null;
  dds::topic::Topic<ddsmetadata::msg::DDSMetaData> topic_ = dds::core::null;
  dds::pub::Publisher publisher_ = dds::core::null;
  dds::pub::DataWriter<ddsmetadata::msg::DDSMetaData> writer_ = dds::core::null;

  std::string topic_name_;
  DDSTransportType dds_transfer_type_;
  size_t max_samples_;

  bool stop_;
};

class CycloneDDSSubListener final : public
      dds::sub::NoOpDataReaderListener<ddsmetadata::msg::DDSMetaData>{
 public:
  CycloneDDSSubListener() : matched_(0) {}
  ~CycloneDDSSubListener() {}
  void on_subscription_matched(
        dds::sub::DataReader<ddsmetadata::msg::DDSMetaData> &reader,
        const dds::core::status::SubscriptionMatchedStatus &info) override;
  std::atomic_uint32_t matched_;
};

using CycloneDDSSubListenerPtr = std::shared_ptr<CycloneDDSSubListener>;

class CycloneDDSSubscriber final : public DDSSubscriber {
 public:
  CycloneDDSSubscriber(const std::string &topic_name,
                       const DDSTransportType &dds_transfer_type,
                       const size_t &max_sample) :
                       stop_(true),
                       topic_name_(topic_name),
                       dds_transfer_type_(dds_transfer_type),
                       max_samples_(max_sample) {}
  ~CycloneDDSSubscriber() override;
  DDSInitErrorType Init();
  void Stop();
  MetaDataPtr Recv(bool keep);
  bool Probe();

 private:
  CycloneDDSSubListenerPtr listener_ = nullptr;
  dds::domain::DomainParticipant participant_ = dds::core::null;
  dds::topic::Topic<ddsmetadata::msg::DDSMetaData> topic_ = dds::core::null;
  dds::sub::Subscriber subscriber_ = dds::core::null;
  dds::sub::DataReader<ddsmetadata::msg::DDSMetaData> reader_ = dds::core::null;
  std::shared_ptr<dds::sub::DataReader<ddsmetadata::msg::DDSMetaData>::Selector>
                 selector_ = nullptr;

  std::string topic_name_;
  DDSTransportType dds_transfer_type_;
  size_t max_samples_;
  bool stop_;
};

}  // namespace message_infrastructure

#endif  // CHANNEL_DDS_CYCLONE_DDS_H_
