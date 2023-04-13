// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#include <channel/dds/cyclone_dds.h>
#include <core/message_infrastructure_logging.h>

#include <vector>
#include <string>

namespace message_infrastructure {

using namespace org::eclipse::cyclonedds;  // NOLINT

void CycloneDDSPubListener::on_offered_incompatible_qos(
  dds::pub::DataWriter<ddsmetadata::msg::DDSMetaData>& writer,
  const dds::core::status::OfferedIncompatibleQosStatus& status) {
  LAVA_LOG_WARN(LOG_DDS,
                "incompatiable qos found, count: %d\n",
                status.total_count());
}

void CycloneDDSPubListener::on_publication_matched(
  dds::pub::DataWriter<ddsmetadata::msg::DDSMetaData>& writer,
  const dds::core::status::PublicationMatchedStatus &info) {
  matched_.store(info.current_count());
  if (info.current_count_change() == 1) {
    LAVA_LOG(LOG_DDS,
             "CycloneDDS DataReader %d matched.\n",
             matched_.load());
  } else if (info.current_count_change() == -1) {
    LAVA_LOG(LOG_DDS,
             "CycloneDDS DataReader unmatched. left:%d\n",
             matched_.load());
  } else {
    LAVA_LOG_ERR("CycloneDDS Publistener MatchedStatus error\n");
  }
}

DDSInitErrorType CycloneDDSPublisher::Init() {
  LAVA_LOG(LOG_DDS, "publisher init\n");
  LAVA_DEBUG(LOG_DDS, "Init CycloneDDS Publisher Successfully, topic name: %s\n",
                  topic_name_.c_str());
  dds_metadata_ = std::make_shared<ddsmetadata::msg::DDSMetaData>();
  if (dds_transfer_type_ != DDSTransportType::DDSUDPv4) {
    LAVA_LOG_WARN(LOG_DDS, "Unsupport Transfer type and will use UDP\n");
  }
  participant_ = dds::domain::DomainParticipant(domain::default_id());
  topic_ = dds::topic::Topic<ddsmetadata::msg::DDSMetaData>(participant_,
                                                            topic_name_);
  publisher_ = dds::pub::Publisher(participant_);
  listener_ = std::make_shared<CycloneDDSPubListener>();
  dds::pub::qos::DataWriterQos wqos = publisher_.default_datawriter_qos();
  wqos << dds::core::policy::History::KeepLast(max_samples_)
       << dds::core::policy::Reliability::Reliable(dds::core::Duration
                                        ::from_secs(HEARTBEAT_PERIOD_SECONDS))
       << dds::core::policy::Durability::Volatile();
  writer_ = dds::pub::DataWriter<ddsmetadata::msg::DDSMetaData>(
            publisher_,
            topic_,
            wqos,
            listener_.get(),
            dds::core::status::StatusMask::all());
  stop_ = false;
  return DDSInitErrorType::DDSNOERR;
}

bool CycloneDDSPublisher::Publish(DataPtr data) {
  LAVA_DEBUG(LOG_DDS,
          "CycloneDDS publisher start publishing topic name = %s, matched:%d\n",
          topic_name_.c_str(), listener_->matched_.load());
  LAVA_DEBUG(LOG_DDS,
             "writer_ matched: %d\n",
             writer_.publication_matched_status().current_count());
  while (writer_.publication_matched_status().current_count() == 0) {
    helper::Sleep();
  }
  LAVA_DEBUG(LOG_DDS, "CycloneDDS publisher find matched reader\n");
  MetaData* metadata = reinterpret_cast<MetaData*>(data.get());
  dds_metadata_->nd(metadata->nd);
  dds_metadata_->type(metadata->type);
  dds_metadata_->elsize(metadata->elsize);
  dds_metadata_->total_size(metadata->total_size);

  memcpy(&dds_metadata_->dims()[0], metadata->dims, sizeof(metadata->dims));
  memcpy(&dds_metadata_->strides()[0],
          metadata->strides,
          sizeof(metadata->strides));
  size_t nbytes = metadata->elsize * metadata->total_size;
  dds_metadata_->mdata(std::vector<uint8_t>(
                  reinterpret_cast<char*>(metadata->mdata),
                  reinterpret_cast<char*>(metadata->mdata) + nbytes));
  LAVA_DEBUG(LOG_DDS, "CycloneDDS publisher copied\n");
  writer_.write(*dds_metadata_.get());
  LAVA_DEBUG(LOG_DDS, "datawriter send the data\n");
  return true;
}

void CycloneDDSPublisher::Stop() {
  LAVA_LOG(LOG_DDS, "Stop CycloneDDS Publisher topic_name%s, waiting unmatched...\n", topic_name_.c_str());
  if (stop_) {
    return;
  }
  while (listener_ != nullptr && listener_->matched_.load() > 0) {
    helper::Sleep();
  }
  if (writer_ != dds::core::null) {
    LAVA_LOG_ERR("pub delete_datawriter\n");
    writer_ = dds::core::null;
  }
  if (publisher_ != dds::core::null) {
    LAVA_LOG_ERR("pub delete_publisher\n");
    publisher_ = dds::core::null;
  }
  if (topic_ != dds::core::null) {
    LAVA_LOG_ERR("pub delete_topic\n");
    topic_ = dds::core::null;
  }
  if (participant_ != dds::core::null) {
    participant_ = dds::core::null;
  }
  stop_ = true;
}
CycloneDDSPublisher::~CycloneDDSPublisher() {
  if (!stop_) {
    Stop();
  }
}

void CycloneDDSSubListener::on_subscription_matched(
    dds::sub::DataReader<ddsmetadata::msg::DDSMetaData> &reader,
    const dds::core::status::SubscriptionMatchedStatus &info) {
  matched_.store(info.current_count());
  if (info.current_count_change() == 1) {
    LAVA_LOG(LOG_DDS,
             "CycloneDDS DataWriter %d matched.\n",
             matched_.load());
  } else if (info.current_count_change() == -1) {
    LAVA_LOG(LOG_DDS,
             "CycloneDDS DataWriter unmatched. left:%d\n",
             matched_.load());
  } else {
    LAVA_LOG_ERR("CycloneDDS Sublistener MatchedStatus error\n");
  }
}
DDSInitErrorType CycloneDDSSubscriber::Init() {
  LAVA_LOG(LOG_DDS, "subscriber init\n");
  LAVA_DEBUG(LOG_DDS, "Init CycloneDDS Subscriber Successfully, topic name: %s\n",
                  topic_name_.c_str());
  if (dds_transfer_type_ != DDSTransportType::DDSUDPv4) {
    LAVA_LOG_WARN(LOG_DDS, "Unsupport Transfer type and will use UDP\n");
  }
  participant_ = dds::domain::DomainParticipant(domain::default_id());
  topic_ = dds::topic::Topic<ddsmetadata::msg::DDSMetaData>(participant_,
                                                            topic_name_);
  subscriber_ = dds::sub::Subscriber(participant_);
  listener_ = std::make_shared<CycloneDDSSubListener>();
  dds::sub::qos::DataReaderQos rqos = subscriber_.default_datareader_qos();
  rqos << dds::core::policy::History::KeepLast(max_samples_)
       << dds::core::policy::Reliability::Reliable(dds::core::Duration
                                        ::from_secs(HEARTBEAT_PERIOD_SECONDS))
       << dds::core::policy::Durability::Volatile();
  dds::core::policy::History history;

  reader_ = dds::sub::DataReader<ddsmetadata::msg::DDSMetaData>(
            subscriber_,
            topic_,
            rqos,
            listener_.get(),
            dds::core::status::StatusMask::all());
  selector_ = std::make_shared<dds::sub::
              DataReader<ddsmetadata::msg::DDSMetaData>::Selector>(reader_);
  selector_->max_samples(1);
  stop_ = false;
  return DDSInitErrorType::DDSNOERR;
}

MetaDataPtr CycloneDDSSubscriber::Recv(bool keep) {
  LAVA_DEBUG(LOG_DDS, "CycloneDDS topic name= %s recving...\n", topic_name_.c_str());
  dds::sub::LoanedSamples<ddsmetadata::msg::DDSMetaData> samples;
  if (keep) {
    LAVA_LOG_ERR(" CycloneDDSSubscriber::Recv keep\n");
    while ((samples = selector_->read()).length() <= 0) {
      helper::Sleep();
    }
  } else {
    LAVA_LOG_ERR(" CycloneDDSSubscriber::Recv\n");
    while ((samples = selector_->take()).length() <= 0) {
      helper::Sleep();
    }
  }

  if (samples.length() != 1) {
    LAVA_LOG_FATAL("Cylones recv %d samples\n", samples.length());
  }
  auto iter = samples.begin();
  if (iter->info().valid()) {
    MetaDataPtr metadata = std::make_shared<MetaData>();
    auto dds_metadata = iter->data();
    metadata->nd = dds_metadata.nd();
    metadata->type = dds_metadata.type();
    metadata->elsize = dds_metadata.elsize();
    metadata->total_size = dds_metadata.total_size();
    memcpy(metadata->dims, dds_metadata.dims().data(), sizeof(metadata->dims));
    memcpy(metadata->strides,
           dds_metadata.strides().data(),
           sizeof(metadata->strides));
    int nbytes = metadata->elsize * metadata->total_size;
    void *ptr = malloc(nbytes);
    memcpy(ptr, dds_metadata.mdata().data(), nbytes);
    metadata->mdata = ptr;
    LAVA_DEBUG(LOG_DDS, "Data Recieved\n");
    return metadata;
  } else {
    LAVA_LOG_ERR("Time out and no data received\n");
  }
  return nullptr;
}

bool CycloneDDSSubscriber::Probe() {
  bool res = false;
  bool res222 = false;
  if ((selector_->read()).length() > 0) {
    res = true;
  }
  if ((selector_->read()).length() > 0) {
    res222 = true;
  }
  LAVA_LOG_ERR("CycloneDDSSubscriber::Probe() res ===%d\n", res);
  LAVA_LOG_ERR("CycloneDDSSubscriber::Probe() res222 ===%d\n", res222);
  return res;
}
void CycloneDDSSubscriber::Stop() {
  if (stop_)
    return;
  LAVA_DEBUG(LOG_DDS, "Subscriber topic name = %s Stop and release...\n", topic_name_.c_str());
  if (listener_ != nullptr && reader_  != dds::core::null) {
    reader_.~DataReader();
    reader_ = dds::core::null;
  }
  if (participant_ != dds::core::null) participant_ = dds::core::null;
  if (subscriber_ != dds::core::null) subscriber_ = dds::core::null;
  if (topic_ != dds::core::null) topic_ = dds::core::null;
  stop_ = true;
}

CycloneDDSSubscriber::~CycloneDDSSubscriber() {
  if (!stop_) {
    Stop();
  }
}

}  // namespace message_infrastructure
