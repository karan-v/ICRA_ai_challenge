#!/usr/bin/env python

import numpy as np
import freenect
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import rospy
import cv2
from std_msgs.msg import Float64
import pyximport
pyximport.install()
from cyt import align
from collections import defaultdict
from io import StringIO
import time
from matplotlib import pyplot as plt
from PIL import Image
global angle
#cap = cv2.VideoCapture(0)

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")


# ## Object detection imports
# Here are the imports from the object detection module.

# In[3]:

from utils import label_map_util

from utils import visualization_utils as vis_util


# # Model preparation 

# ## Variables
# 
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.  
# 
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

# In[4]:
def get_video():
    array,_ = freenect.sync_get_video(0,freenect.VIDEO_IR_10BIT)
    return array

def correct_points(v2,u2,z2):
    f2u = 527.97431353
    f2v = 523.22880991
    c2u = 312.66606295
    c2v = 255.35571034
    y2 = ((u2-c2u)*z2)/f2u
    x2 = ((v2-c2v)*z2)/f2v
    return y2,x2




# What model to download.
#MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_NAME = 'mac_n_cheese_inference_graph'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = '/home/karan/models/research/object_detection/target_inference_graph' + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
#PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
PATH_TO_LABELS = os.path.join('training', 'object-detection.pbtxt')


NUM_CLASSES = 90


### load image

#testing_img = cv2.imread('test_images/image3.jpg')
#testing_img = cv2.cvtColor(testing_img,cv2.COLOR_BGR2RGB)

# In[6]:

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

# In[7]:

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)



IMAGE_SIZE = (12, 8)


# In[10]:

with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    while True:
      #ret, image_np = cap.read()
      #image_np = testing_img
      image_np,_ = freenect.sync_get_video()
      #image_np = cv2.resize(image_np, (480,640))
      image_np = cv2.cvtColor(image_np,cv2.COLOR_BGR2RGB)
      depth, timestamp = freenect.sync_get_depth()
      depth1 = depth.astype(np.uint8)
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      # Each box represents a part of the image where a particular object was detected.
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      # Each score represent how level of confidence for each of the objects.
      # Score is shown on the result image, together with the class label.
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      # Actual detection.
      (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      # Visualization of the results of a detection.
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=8)
      height, width = image_np.shape[:2]
      ymin = boxes[0][0][0]*height
      xmin = boxes[0][0][1]*width
      ymax = boxes[0][0][2]*height
      xmax = boxes[0][0][3]*width
      #cv2.circle(image_np,(int((xmin+xmax)/2.0),int((ymin+ymax)/2.0)), 10, (0,0,255), -1)
      #cv2.circle(image_np,(int(image_np.shape[1]/2.0),int(image_np.shape[0]/2.0)), 10, (255,0,0), -1)
      p1 = np.array([[int((ymin+ymax)/2.0)],[int((xmin+xmax)/2.0)]])
      p2 = np.array([[int(image_np.shape[0]/2.0)],[int(image_np.shape[1]/2.0)]])
      
      cv2.line(image_np,(0,240),(640,240),(0,255,255),4)
      cv2.line(image_np,(320,0),(320,480),(123,23,111),4)
      depth2 = np.zeros((480,640))
      new_depth = align(depth,depth2)
      new_depth = np.asarray(new_depth)
      raw_aligned_depth = new_depth.copy()
      raw_aligned_depth = np.float32(raw_aligned_depth)
      z2 = 1.0/(-0.00307 * raw_aligned_depth[int((ymin+ymax)/2.0)][int((xmin+xmax)/2.0)] + 3.33)
      n,m = correct_points(p1[0],p1[1],z2)
      p3 = np.array([[n],[m]])
      z1 = 1.0/(-0.00307 * raw_aligned_depth[int(image_np.shape[0]/2)][int(image_np.shape[1]/2)] + 3.33)
      n,m = correct_points(p2[0],p2[1],z1)
      p4 = np.array([[n],[m]])
      new_depth = new_depth.astype(np.uint8)
      xmax = np.sin((30*np.pi)/180)*z2
      x1 = (np.sqrt(p3[0]**2+p3[1]**2)/xmax)*z2
      angle = (np.tanh(np.abs(x1)/z2))*(180/np.pi)
      print(angle)
      cv2.circle(image_np,(int(312.66606295),int(255.35571034)),10,(0,0,255),-1)
      cv2.imshow('object detection1', depth1)
      cv2.imshow('object detection', image_np)
      if cv2.waitKey(25) and 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

"""def talker():
    
    pub = rospy.Publisher('angle', Float64, queue_size=10)
    rospy.init_node('talker',anonymous=True)
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        with detection_graph.as_default():
	  with tf.Session(graph=detection_graph) as sess:
	    while True:
	      #ret, image_np = cap.read()
	      #image_np = testing_img
	      image_np,_ = freenect.sync_get_video()
	      #image_np = cv2.resize(image_np, (480,640))
	      image_np = cv2.cvtColor(image_np,cv2.COLOR_BGR2RGB)
	      depth, timestamp = freenect.sync_get_depth()
	      depth1 = depth.astype(np.uint8)
	      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
	      image_np_expanded = np.expand_dims(image_np, axis=0)
	      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
	      # Each box represents a part of the image where a particular object was detected.
	      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
	      # Each score represent how level of confidence for each of the objects.
	      # Score is shown on the result image, together with the class label.
	      scores = detection_graph.get_tensor_by_name('detection_scores:0')
	      classes = detection_graph.get_tensor_by_name('detection_classes:0')
	      num_detections = detection_graph.get_tensor_by_name('num_detections:0')
	      # Actual detection.
	      (boxes, scores, classes, num_detections) = sess.run(
		  [boxes, scores, classes, num_detections],
		  feed_dict={image_tensor: image_np_expanded})
	      # Visualization of the results of a detection.
	      vis_util.visualize_boxes_and_labels_on_image_array(
		  image_np,
		  np.squeeze(boxes),
		  np.squeeze(classes).astype(np.int32),
		  np.squeeze(scores),
		  category_index,
		  use_normalized_coordinates=True,
		  line_thickness=8)
	      height, width = image_np.shape[:2]
	      ymin = boxes[0][0][0]*height
	      xmin = boxes[0][0][1]*width
	      ymax = boxes[0][0][2]*height
	      xmax = boxes[0][0][3]*width
	      cv2.circle(image_np,(int((xmin+xmax)/2.0),int((ymin+ymax)/2.0)), 10, (0,0,255), -1)
	      cv2.circle(image_np,(int(image_np.shape[1]/2.0),int(image_np.shape[0]/2.0)), 10, (255,0,0), -1)
	      p1 = np.array([[int((ymin+ymax)/2.0)],[int((xmin+xmax)/2.0)]])
	      p2 = np.array([[int(image_np.shape[0]/2.0)],[int(image_np.shape[1]/2.0)]])
	      
	      cv2.line(image_np,(0,240),(640,240),(0,255,255),4)
	      cv2.line(image_np,(320,0),(320,480),(123,23,111),4)
	      depth2 = np.zeros((480,640))
	      new_depth = align(depth,depth2)
	      new_depth = np.asarray(new_depth)
	      raw_aligned_depth = new_depth.copy()
	      raw_aligned_depth = np.float32(raw_aligned_depth)
	 
	      z2 = 1.0/(-0.00307 * raw_aligned_depth[int((ymin+ymax)/2.0)][int((xmin+xmax)/2.0)] + 3.33)
	      
	      n,m = correct_points(p1[0],p1[1],z2)
	      p3 = np.array([[n],[m]])
	      z1 = 1.0/(-0.00307 * raw_aligned_depth[int(image_np.shape[0]/2)][int(image_np.shape[1]/2)] + 3.33)
	      n,m = correct_points(p2[0],p2[1],z1)
	      p4 = np.array([[n],[m]])
	      
	      new_depth = new_depth.astype(np.uint8)
	      
	      xmax = np.sin((30*np.pi)/180)*z2
	      x1 = (np.sqrt(p3[0]**2+p3[1]**2)/xmax)*z2
	      angle = (np.sinh(np.abs(x1)/np.sqrt(x1*x1 + z2*z2)))*(180/np.pi)
	      #angle = (np.sinh(np.abs(x1)/z2))*(180/np.pi)
              rospy.loginfo(angle)
              pub.publish(angle)
	      cv2.circle(image_np,(int(312.66606295),int(255.35571034)),10,(0,0,255),-1)
	      cv2.imshow('object detection1', depth1)
	      cv2.imshow('object detection', image_np)
	      if cv2.waitKey(25) and 0xFF == ord('q'):
		cv2.destroyAllWindows()
		break

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass"""
