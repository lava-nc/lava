// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#include <message_infrastructure/csrc/channel/dds/cyclone_dds.h>
#include <message_infrastructure/csrc/core/message_infrastructure_logging.h>

#include <vector>
#include <string>

namespace message_infrastructure {

using namespace org::eclipse::cyclonedds;

void CycloneDDSPubListener::on_offered_incompatible_qos(
  dds::pub::DataWriter<DDSMetaData>& writer,
  const dds::core::status::OfferedIncompatibleQosStatus& status) {
  LAVA_LOG_WARN(LOG_DDS, "incompatiable qos found, count: %d\n", status.total_count());
}

void CycloneDDSPubListener::on_publication_matched(
  dds::pub::DataWriter<DDSMetaData>& writer,
  const dds::core::status::PublicationMatchedStatus &info) {
  if (info.current_count_change() == 1) {
    matched_.store(info.current_count(), std::memory_order_release);
    LAVA_LOG(LOG_DDS, "CycloneDDS DataReader %d matched.\n", matched_.load(std::memory_order_release));
  } else if (info.current_count_change() == -1) {
    matched_.store(info.current_count(), std::memory_order_release);
    LAVA_LOG(LOG_DDS, "CycloneDDS DataReader unmatched. left:%d\n", matched_.load(std::memory_order_release));
  } else {
    LAVA_LOG_ERR("CycloneDDS Publistener MatchedStatus error\n");
  }
}

int CycloneDDSPublisher::Init() {
  LAVA_LOG(LOG_DDS, "publisher init\n");
  dds_metadata_ = std::make_shared<DDSMetaData>();
  // cyclone participantqos only has usedata and factory policy.
  participant_ = dds::domain::DomainParticipant(domain::default_id());
  topic_ = dds::topic::Topic<DDSMetaData>(participant_, topic_name_);
  publisher_ = dds::pub::Publisher(participant_);
  listener_ = std::make_shared<CycloneDDSPubListener>();
  dds::pub::qos::DataWriterQos wqos = publisher_.default_datawriter_qos();
  wqos << dds::core::policy::History::KeepLast(32)
       << dds::core::policy::Reliability::Reliable(dds::core::Duration::from_secs(5))
       << dds::core::policy::Durability::Volatile(); // volatile for shm
  writer_ = dds::pub::DataWriter<DDSMetaData>(publisher_,
                                              topic_,
                                              wqos,
                                              listener_.get(),
                                              dds::core::status::StatusMask::all());
  stop_ = false;
  return 0;
}

bool CycloneDDSPublisher::Publish(MetaDataPtr metadata) {
  LAVA_DEBUG(LOG_DDS, "CycloneDDS publisher start publishing, matched:%d\n", listener_->matched_.load(std::memory_order_release));
  LAVA_DEBUG(LOG_DDS, "writer_ matched: %d\n", writer_.publication_matched_status().current_count());
  while (listener_->matched_.load(std::memory_order_release) == 0) {
    helper::Sleep();
  }
  LAVA_DEBUG(LOG_DDS, "CycloneDDS publisher find matched reader\n");
  dds_metadata_->nd(metadata->nd);
  dds_metadata_->type(metadata->type);
  dds_metadata_->elsize(metadata->elsize);
  dds_metadata_->total_size(metadata->total_size);

  memcpy(&dds_metadata_->dims()[0], metadata->dims, sizeof(metadata->dims));
  memcpy(&dds_metadata_->strides()[0],
          metadata->strides,
          sizeof(metadata->strides));
  size_t nbytes = metadata->elsize * metadata->total_size;
  dds_metadata_->mdata(std::vector<char>(
                  reinterpret_cast<char*>(metadata->mdata),
                  reinterpret_cast<char*>(metadata->mdata) + nbytes));
  LAVA_DEBUG(LOG_DDS, "CycloneDDS publisher copied\n");
  writer_.write(*dds_metadata_.get());
  LAVA_DEBUG(LOG_DDS, "datawriter send the data\n");
  return true;
}

void CycloneDDSPublisher::Stop() {
  LAVA_LOG(LOG_DDS, "Stop CycloneDDS Publisher, waiting unmatched...\n");
  if (stop_) {
    return;
  }
  while (listener_->matched_ > 0) {
    helper::Sleep();
  }
  try {
    writer_.~DataWriter();
    participant_ = dds::core::null;
    publisher_ = dds::core::null;
    topic_ = dds::core::null;
    writer_ = dds::core::null;
  } catch (const dds::core::Exception& e) {
    std::cerr << "=== [Publisher] Exception: " << e.what() << std::endl;
  }
  stop_ = true;
}
CycloneDDSPublisher::~CycloneDDSPublisher() {
  if(!stop_) {
    Stop();
  }
}

void CycloneDDSSubListener::on_subscription_matched(
    dds::sub::DataReader<DDSMetaData> &reader,
    const dds::core::status::SubscriptionMatchedStatus &info) {
  if (info.current_count_change() == 1) {
    matched_.store(info.current_count(), std::memory_order_release);
    LAVA_LOG(LOG_DDS, "CycloneDDS DataWriter %d matched.\n", info.current_count());
  } else if (info.current_count_change() == -1) {
    matched_.store(info.current_count(), std::memory_order_release);
    LAVA_LOG(LOG_DDS, "CycloneDDS DataWriter unmatched. left:%d\n", info.current_count());
  } else {
    LAVA_LOG_ERR("CycloneDDS Sublistener MatchedStatus error\n");
  }
}
int CycloneDDSSubscriber::Init() {
  LAVA_LOG(LOG_DDS, "subscriber init\n");
  participant_ = dds::domain::DomainParticipant(domain::default_id());
  topic_ = dds::topic::Topic<DDSMetaData>(participant_, topic_name_);
  subscriber_ = dds::sub::Subscriber(participant_);
  listener_ = std::make_shared<CycloneDDSSubListener>();
  dds::sub::qos::DataReaderQos rqos = subscriber_.default_datareader_qos();
  rqos << dds::core::policy::History::KeepLast(32)
       << dds::core::policy::Reliability::Reliable(dds::core::Duration::from_secs(5))
       << dds::core::policy::Durability::Volatile();
  dds::core::policy::History history;

  reader_ = dds::sub::DataReader<DDSMetaData>(subscriber_,
                                              topic_,
                                              rqos,
                                              listener_.get(),
                                              dds::core::status::StatusMask::all());
  selector_ = std::make_shared<dds::sub::DataReader<DDSMetaData>::Selector>(reader_);
  selector_->max_samples(1);
  stop_ = false;
  return 0;
}

MetaDataPtr CycloneDDSSubscriber::Recv(bool keep) {
  LAVA_DEBUG(LOG_DDS, "CycloneDDS recving...\n");
  // while(listener_->matched_ <= 0) {
  //   helper::Sleep();
  // }
  // LAVA_DEBUG(LOG_DDS, "CycloneDDS Writer Matched.\n");
  dds::sub::LoanedSamples<DDSMetaData> samples;
  if (keep) {
    while ((samples = selector_->read()).length() <= 0) {
      helper::Sleep();
    }
  } else {
    while ((samples = selector_->take()).length() <= 0) {
      helper::Sleep();
    }
  }

  if (samples.length() != 1) {
    LAVA_LOG_ERR("FATAL: Cylones recv %d samples\n", samples.length());
    exit(-1);
  }
  auto iter = samples.begin();
  if(iter->info().valid()) {
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
    LAVA_LOG_ERR("time out and no data received\n");
  }
  return nullptr;
}

void CycloneDDSSubscriber::Stop() {
  if (stop_)
    return;
  LAVA_DEBUG(LOG_DDS, "Subscriber Stop and release...\n");
  try {
    reader_.~DataReader();
    participant_ = dds::core::null;
    subscriber_ = dds::core::null;
    topic_ = dds::core::null;
    reader_ = dds::core::null;
  } catch (const dds::core::Exception& e) {
    std::cerr << "=== [Publisher] Exception: " << e.what() << std::endl;
  }
  stop_ = true;
}

CycloneDDSSubscriber::~CycloneDDSSubscriber() {
  if (!stop_) {
    Stop();
  }
}
}  // namespace message_infrastructure