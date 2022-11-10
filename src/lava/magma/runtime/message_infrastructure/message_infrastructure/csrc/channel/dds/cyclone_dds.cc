// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#include <message_infrastructure/csrc/channel/dds/cyclone_dds.h>
#include <message_infrastructure/csrc/core/message_infrastructure_logging.h>

#include <vector>
#include <string>

namespace message_infrastructure {

using namespace org::eclipse::cyclonedds;

int CycloneDDSPublisher::Init() {

  dds_metadata_ = std::make_shared<DDSMetaData>();
  participant_ = dds::domain::DomainParticipant(domain::default_id());
  topic_ = dds::topic::Topic<DDSMetaData>(participant_, "DDSMetaData_TOPIC");
  publisher_ = dds::pub::Publisher(participant_);
  writer_ = dds::pub::DataWriter<DDSMetaData>(publisher_, topic_);
  stop_ = false;

  return 0;
}

bool CycloneDDSPublisher::Publish(MetaDataPtr metadata) {
  LAVA_DEBUG(LOG_DDS, "CycloneDDS publisher start publishing...\n");
  dds_metadata_->nd(metadata->nd);
  dds_metadata_->type(metadata->type);
  dds_metadata_->elsize(metadata->elsize);
  dds_metadata_->total_size(metadata->total_size);
  printf("copy array...\n");
  memcpy(&dds_metadata_->dims()[0], metadata->dims, sizeof(metadata->dims));
  memcpy(&dds_metadata_->strides()[0],
          metadata->strides,
          sizeof(metadata->strides));
  size_t nbytes = metadata->elsize * metadata->total_size;
  dds_metadata_->mdata(std::vector<char>(
                  reinterpret_cast<char*>(metadata->mdata),
                  reinterpret_cast<char*>(metadata->mdata) + nbytes));
  LAVA_DEBUG(LOG_DDS, "CycloneDDS publisher copied\n");
  while (writer_->publication_matched_status().current_count() == 0) {
    helper::Sleep();
  }
  LAVA_DEBUG(LOG_DDS, "CycloneDDS datawriter find subscriber\n");
  writer_.write(*dds_metadata_.get());
  LAVA_DEBUG(LOG_DDS, "datawriter send the data\n");
  return true;
}

void CycloneDDSPublisher::Stop() {
  LAVA_LOG(LOG_DDS, "Stop CycloneDDS Publisher, waiting unmatched...\n");
  while (writer_->publication_matched_status().current_count() > 0) {
    helper::Sleep();
  }
  // TODO: Delete
}
CycloneDDSPublisher::~CycloneDDSPublisher() {
}

int CycloneDDSSubscriber::Init() {
  printf("subscriber init\n");
  participant_ = dds::domain::DomainParticipant(domain::default_id());
  topic_ = dds::topic::Topic<DDSMetaData>(participant_, "DDSMetaData_TOPIC");
  subscriber_ = dds::sub::Subscriber(participant_);
  reader_ = dds::sub::DataReader<DDSMetaData>(subscriber_, topic_);
  selector_ = std::make_shared<dds::sub::DataReader<DDSMetaData>::Selector>(reader_);
  stop_ = false;
  return 0;
}

MetaDataPtr CycloneDDSSubscriber::Recv(bool keep) {
  LAVA_LOG(LOG_DDS, "CycloneDDS recving...\n");
  dds::sub::LoanedSamples<DDSMetaData> samples;
  // dds::sub::LoanedSamples<DDSMetaData> samples;
  // do {
  //   samples = reader_->read();
  // } while (samples.length() <= 0);

  // dds::sub::DataReader<DDSMetaData>::Selector reader(reader_);
  // reader.max_samples(1);
  
  do{
    samples = selector_->take();
  } while (samples.length() <= 0);

  printf("%d\n", samples.length());
  
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
  LAVA_DEBUG(LOG_DDS, "Recieved OK\n");
  return nullptr;
}

void CycloneDDSSubscriber::Stop() {
  LAVA_DEBUG(LOG_DDS, "Subscriber Stop and release...\n");
}

CycloneDDSSubscriber::~CycloneDDSSubscriber() {

}
}  // namespace message_infrastructure