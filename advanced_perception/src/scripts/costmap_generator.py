#!/usr/bin/env python3

from PIL import Image
import rospy
import numpy as np
import ros_numpy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import Image as ros_img
import cv2

# ==============================================================================
#                             SCALE_TO_255
# ==============================================================================
pub = None

def point_cloud_callback(msg):
    # Convert PointCloud2 message to a numpy array
    point_cloud_array = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(msg)
    # print("converted successfully")
    birds_eye_point_cloud(point_cloud_array)
   

def scale_to_255(a, min, max, dtype=np.uint8):
    """ Scales an array of values from specified min, max range to 0-255
        Optionally specify the data type of the output (default is uint8)
    """
    return (((a - min) / float(max - min)) * 255).astype(dtype)

def birds_eye_point_cloud(points,
                          side_range=(-20, 10),  #(top_y limit , lower_y_limit)
                          fwd_range=(-3,3),
                          res=0.13,
                          min_height = -2.73,
                          max_height = 1.27,
                          saveto="map.png"):
    """ Creates an 2D birds eye view representation of the point cloud data.
        You can optionally save the image to specified filename.

    Args:
        points:     (numpy array)
                    N rows of points data
                    Each point should be specified by at least 3 elements x,y,z
        side_range: (tuple of two floats)
                    (-left, right) in metres
                    left and right limits of rectangle to look at.
        fwd_range:  (tuple of two floats)
                    (-behind, front) in metres
                    back and front limits of rectangle to look at.
        res:        (float) desired resolution in metres to use
                    Each output pixel will represent an square region res x res
                    in size.
        min_height:  (float)(default=-2.73)
                    Used to truncate height values to this minumum height
                    relative to the sensor (in metres).
                    The default is set to -2.73, which is 1 metre below a flat
                    road surface given the configuration in the kitti dataset.
        max_height: (float)(default=1.27)
                    Used to truncate height values to this maximum height
                    relative to the sensor (in metres).
                    The default is set to 1.27, which is 3m above a flat road
                    surface given the configuration in the kitti dataset.
        saveto:     (str or None)(default=None)
                    Filename to save the image as.
                    If None, then it just displays the image.
    """
    x_lidar = points[:, 0]
    # print(len(points) , len(x_lidar))
    y_lidar = points[:, 2]
    z_lidar = points[:, 1]
    # r_lidar = points[:, 3]  # Reflectance

    # INDICES FILTER - of values within the desired rectangle
    # Note left side is positive y axis in LIDAR coordinates
    ff = np.logical_and((x_lidar > fwd_range[0]), (x_lidar < fwd_range[1]))
    ss = np.logical_and((y_lidar > -side_range[1]), (y_lidar < -side_range[0]))
    indices = np.argwhere(np.logical_and(ff,ss)).flatten()

    # CONVERT TO PIXEL POSITION VALUES - Based on resolution
    x_img = (-y_lidar[indices]/res).astype(np.int32) # x axis is -y in LIDAR
    y_img = (x_lidar[indices]/res).astype(np.int32)  # y axis is -x in LIDAR
                                                     # will be inverted later

    # SHIFT PIXELS TO HAVE MINIMUM BE (0,0)
    # floor used to prevent issues with -ve vals rounding upwards
    x_img -= int(np.floor(side_range[0]/res))
    y_img -= int(np.floor(fwd_range[0]/res))

    # CLIP HEIGHT VALUES - to between min and max heights
    pixel_values = np.clip(a = z_lidar[indices],
                           a_min=min_height,
                           a_max=max_height)

    # RESCALE THE HEIGHT VALUES - to be between the range 0-255
    pixel_values  = scale_to_255(pixel_values, min=min_height, max=max_height)

    # FILL PIXEL VALUES IN IMAGE ARRAY
    x_max = int((side_range[1] - side_range[0])/res)
    y_max = int((fwd_range[1] - fwd_range[0])/res)
    im = np.zeros([y_max, x_max], dtype=np.uint8)
    im[-y_img, x_img] = pixel_values # -y because images start from top left
    im = cv2.bitwise_not(im) 
    # print(im.shape)
    # cv2.imshow("Frame",im)
    # cv2.waitKey(1)

    # Convert from numpy array to a PIL image
    im = Image.fromarray(im)
    im = im.rotate(270)
    im = im.convert('RGB')
    output = ros_img()
    output.header.stamp = rospy.Time.now()
    output.height = im.height
    output.width = im.width
    output.encoding = "rgb8"
    output.is_bigendian = False
    output.step = 3 * im.width
    output.data = np.array(im).tobytes()
    global pub
    pub.publish(output)
  
def main():
    global pub 
    rospy.init_node('costmap_generator')
    rospy.Subscriber("/filtered_cloud", PointCloud2, point_cloud_callback)
    pub = rospy.Publisher('/costmap',ros_img, queue_size=1)

    # Spin ROS
    rospy.spin()

if __name__ == '__main__':
    main()