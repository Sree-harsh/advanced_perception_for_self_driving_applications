#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

#include <pcl/io/pcd_io.h>
#include <pcl/filters/crop_box.h>
#include <pcl_conversions/pcl_conversions.h>

#define RATE 15

ros::Publisher pub;
ros::Rate *loop_rate;

pcl::PointCloud<pcl::PointXYZRGB>::Ptr cuboidCloud(new pcl::PointCloud<pcl::PointXYZRGB>);

std::default_random_engine gen;
std::uniform_real_distribution<double> neg(-2, 0);
std::uniform_real_distribution<double> pos(0, 2);

void pointcloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg) {\

  // std::cout << "pointCloudCallback called\n";
  auto start = std::chrono::high_resolution_clock::now();

  // Convert PointCloud2 message to pcl::PointCloud
  pcl::PointCloud<pcl::PointXYZRGB> pcl_cloud;
  sensor_msgs::PointCloud2 output;
  pcl::fromROSMsg(*msg, pcl_cloud);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr filteredCloud(new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointIndices::Ptr indices(new pcl::PointIndices);
  for (size_t i = 0; i < pcl_cloud.size(); ++i)
  {
    pcl::PointXYZRGB point = pcl_cloud[i];
    if (point.r != 0 || point.g != 0 || point.b != 0)
    {
      filteredCloud->push_back(point);
      indices->indices.push_back(i);
    }
  }

  // Save pcl::PointCloud to PCD file
  // pcl::io::savePCDFileASCII("cuboid.pcd", *cuboidCloud);
  // ROS_INFO("PCD file saved.");

  // pcl::toROSMsg(*filteredCloud + *cuboidCloud, output); // Code for concatenation of two point clouds if needed
  pcl::toROSMsg(*filteredCloud, output);
  output.header.frame_id = "map_new"; //Change frame_id 
  pub.publish(output);
  cuboidCloud->clear();

  //Fps calculation code

  auto stop = std::chrono::high_resolution_clock::now();
  std::cout << std::setprecision(3)
            << "Running at "
            << 1e9 / std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count()
            << "fps\n";
            
  loop_rate->sleep();
}



int main(int argc, char** argv)
{
  ros::init(argc, argv, "lane_filterer"); //Node initialization
  ros::NodeHandle nh;
  
  loop_rate = new ros::Rate(RATE); 

  pub = nh.advertise<sensor_msgs::PointCloud2>("/filtered_cloud", 1);
  // ros::Subscriber sub = nh.subscribe<sensor_msgs::PointCloud2>("/camera/depth/points_marked_potholes", 1, pointcloudCallback);
  ros::Subscriber sub = nh.subscribe<sensor_msgs::PointCloud2>("/camera/depth/points_marked", 1, pointcloudCallback);


  ros::spin();

  return 0;
}





