#!/usr/bin/env python
#
# COMPUTER VISION FOR ROBOTICS
# PLANE SEGMENTATION USING RANSAC AND PCA
#
# Instructions:
# Complete the code to find a plane given a point cloud
# You can assume a predefined plane orientation.
# Return the plane model in the point-normal form, inliers and outliers
#

import rospy
import cv2
import numpy
import Utils
import tf
from vision_msgs.srv import FindPlanes, FindPlanesResponse

def find_plane(xyz):
    inliers = numpy.zeros(xyz.shape)
    outliers = numpy.zeros(xyz.shape)
    center = [0,0,0]
    normal = [0,0,0]
    #
    # TODO:
    # Find a plane given the point cloud xyz
    # Return the plane in the point-normal form, inliers and outliers
    #
    return center, normal, inliers, outliers

def callback_find_plane(req):
    print("Trying to find plane...")
    transformer = tf.TransformerROS()
    bgr, xyz = Utils.point_cloud_2_to_cv_images(req.point_cloud)
    center, normal, inliers, outliers = find_plane(xyz)
    resp = FindPlanesResponse()
    resp.header.frame_id = req.point_cloud.header.frame_id
    resp.header.stamp    = req.point_cloud.header.stamp
    resp.center.x = center[0]
    resp.center.y = center[1]
    resp.center.z = center[2]
    resp.normal.x = normal[0]
    resp.normal.y = normal[1]
    resp.normal.z = normal[2]
    print("Center:")
    print(resp.center)
    print("Normal")
    print(resp.normal)
    

def main():
    print("INITIALIZING PLAN FINDER..")
    rospy.init_node("plane_finder")
    rospy.Service("/vision/line_finder/find_horizontal_plane_ransac", FindPlanes, callback_find_plane)
    rospy.spin()

if __name__ == '__main__':
    main()
    
