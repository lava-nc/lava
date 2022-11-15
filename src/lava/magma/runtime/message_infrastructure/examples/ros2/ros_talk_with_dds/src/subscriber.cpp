#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "ddsmetadata/msg/dds_meta_data.hpp"
using std::placeholders::_1;

class MinimalSubscriber : public rclcpp::Node
{
public:
  MinimalSubscriber()
  : Node("minimal_subscriber")
  {
    subscription_ = this->create_subscription<ddsmetadata::msg::DDSMetaData>(
      "dds_topic", rclcpp::SensorDataQoS(), std::bind(&MinimalSubscriber::topic_callback, this, _1));
  }

private:
  void topic_callback(const ddsmetadata::msg::DDSMetaData::SharedPtr metadata) const
  {
    unsigned char* ptr = reinterpret_cast<unsigned char*> (malloc(sizeof(int64_t)));
    for(int i = 0; i < 8; i++)
            *(ptr + i) = metadata->mdata[i];
    RCLCPP_INFO(this->get_logger(), "ROS2 heard: '%Ld'", *reinterpret_cast<int64_t*>(ptr));
  }
  rclcpp::Subscription<ddsmetadata::msg::DDSMetaData>::SharedPtr subscription_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MinimalSubscriber>());
  rclcpp::shutdown();
  return 0;
}
