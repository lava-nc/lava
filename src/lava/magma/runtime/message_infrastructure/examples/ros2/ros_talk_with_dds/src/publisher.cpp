// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#include <chrono>
#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "ddsmetadata/msg/dds_meta_data.hpp"

using namespace std::chrono_literals;

#define MAX_ARRAY_DIMS (5)

struct MetaData {
  int64_t nd;
  int64_t type;
  int64_t elsize;
  int64_t total_size;
  int64_t dims[MAX_ARRAY_DIMS] = {0};
  int64_t strides[MAX_ARRAY_DIMS] = {0};
  void* mdata;
};

using MetaDataPtr = std::shared_ptr<MetaData>;

class MinimalPublisher : public rclcpp::Node
{
public:
  MinimalPublisher()
  : Node("minimal_publisher"), count_(0)
  {
    metadata = std::make_shared<MetaData>();
    metadata->nd = 1;
    metadata->type = 7;
    metadata->elsize = 8;
    metadata->total_size = 1;
    metadata->dims[0] = 1;
    metadata->strides[0] = 1;
    metadata->mdata = reinterpret_cast<char*> (malloc(sizeof(int64_t)));
    *reinterpret_cast<int64_t*>(metadata->mdata) = 0;

    publisher_ = this->create_publisher<ddsmetadata::msg::DDSMetaData>("dds_topic", rclcpp::SystemDefaultsQoS());
    timer_ = this->create_wall_timer(
      500ms, std::bind(&MinimalPublisher::timer_callback, this));
  }

private:
  void timer_callback()
  {
    auto message = ddsmetadata::msg::DDSMetaData();
    message.nd = metadata->nd;
    message.type = metadata->type;
    message.elsize = metadata->elsize;
    message.total_size = metadata->total_size;
    for(int i = 0; i < MAX_ARRAY_DIMS; i++) {
      message.dims[i] = metadata->dims[i];
      message.strides[i] = metadata->strides[i];
    }
    *reinterpret_cast<int64_t*>(metadata->mdata) = count_;
    message.mdata = std::vector<unsigned char>(
                  reinterpret_cast<unsigned char*>(metadata->mdata),
                  reinterpret_cast<unsigned char*>(metadata->mdata) + metadata->elsize * metadata->total_size);
    
    RCLCPP_INFO(this->get_logger(), "ROS2 publishing: '%d'", count_++);
    publisher_->publish(message);
  }
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<ddsmetadata::msg::DDSMetaData>::SharedPtr publisher_;
  size_t count_;
  MetaDataPtr metadata;
  rmw_publisher_t publisher;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MinimalPublisher>());
  rclcpp::shutdown();
  return 0;
}