#!/usr/bin/env python
import rospy
import cv2
import numpy
from vision_msgs.srv import RecognizeObject, RecognizeObjects, RecognizeObjectsResponse
from vision_msgs.msg import VisionObject
from cv_bridge import CvBridge
from yolov5.utils.torch_utils import select_device
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import *
import torch
import ros_numpy

def callback_recognize_objects(req):
    # to call by parameter: rosrun object_classification obj_classification_node.py _model:=/home/joel/Repositories/ComputerVisionForRobotics-2024-2/catkin_ws/src/vision/object_classification/src/weights/TaRJustv3_ycb.pt
    global device, model, min_confidence, result_img,process_img
    print("Received request img encodeing " + req.image.encoding)
    bridge = CvBridge()
    img = bridge.imgmsg_to_cv2(req.image)   
    cloud = ros_numpy.numpify(req.point_cloud)
    process_img = numpy.copy(img)
    w = process_img.shape[0]
    h = process_img.shape[1] 
    mask_acum = np.zeros((w,h,1),dtype=np.uint8)
    mask_final = np.zeros(process_img.shape,dtype=np.uint8)
    print(cloud.shape)
    result_img = numpy.copy(img)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img).to(device) # RGB IMAGE TENSOR (TORCH)
    img = img / 255.0                              #NORMALIZE
    img = img.unsqueeze(0)                        # ADD DIMENSION FOR TENSOR ( BATCH)
    img = torch.moveaxis(img,3,1)                  #Channel order for YOLO
    pred = model(img, augment=False)[0]
    pred = non_max_suppression(pred)  # IOU
    

    resp = RecognizeObjectsResponse()
    for det in pred:
        for x0,y0,x1,y1, conf, cls in (det):# Model Result is bounding box  confidence  and class
            confidence = conf.cpu().tolist()
            if confidence > min_confidence:
                x0 = int(x0.cpu().tolist())
                y0 = int(y0.cpu().tolist())
                x1 = int(x1.cpu().tolist())
                y1 = int(y1.cpu().tolist())
                name = model.names[int(cls.cpu().tolist())] 
                result_img=cv2.rectangle(result_img ,(x0,y0), (x1,y1),  (0, 255, 0), 2)
                mask_black = np.zeros(process_img.shape,dtype=np.uint8)
                mask = cv2.rectangle(mask_black ,(x0-1,y0-1), (x1+1,y1+1),  (255, 255, 255), -1)
                gray  = cv2.cvtColor(process_img, cv2.COLOR_BGR2GRAY)
                canny = cv2.Canny(gray, 50, 254)
                canny = cv2.bitwise_and(canny,mask[:,:,0])
                kernel = np.ones((15, 15), np.uint8)
                dilated = cv2.dilate(canny,kernel)
                eroded = cv2.erode(dilated,kernel)
                contours, hierarchy = cv2.findContours(eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                for c in contours:
                   poly = cv2.fillConvexPoly(eroded, points=c, color=(255,255,255))
                   eroded = cv2.bitwise_or(eroded,poly)                   
                mask_acum = cv2.bitwise_or(mask_acum,eroded)
                result_img= cv2.putText(result_img, name+" "+str(confidence),(x0,y0-2),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
                visobj = VisionObject()
                visobj.id = name
                visobj.confidence = confidence
                resp.recog_objects.append(visobj)
    mask_final[:,:,0] = mask_acum
    mask_final[:,:,1] = mask_acum
    mask_final[:,:,2] = mask_acum
    process_img = cv2.bitwise_and(mask_final,process_img)
    return resp
    

def main():
    global device, model, min_confidence, result_img, process_img
    print("INITIALIZING OsBJECT CLASSIFICATION in an Oscarly manner...")
    rospy.init_node("object_classification")
    model_name = rospy.get_param("~model", "ycb.pt")
    min_confidence = rospy.get_param("~min_confidence", 0.5)
    loop = rospy.Rate(10)

    device = select_device('')
    print("ObjClassification.->Loading model: " + model_name)
    model  = attempt_load(model_name, device)
    print("ObjClassification.->Loaded model")
    result_img = numpy.zeros((512, 512, 3), numpy.uint8)
    process_img = numpy.zeros((512, 512, 3), numpy.uint8)

    rospy.Service("/vision/obj_reco/detect_and_recognize_objects", RecognizeObjects, callback_recognize_objects)
    while not rospy.is_shutdown():
        cv2.imshow("YOLO - Recognition Result", result_img)
        cv2.imshow("Processing", process_img)
        cv2.waitKey(10)
        loop.sleep()

if __name__ == "__main__":
    main()
