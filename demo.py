from TinyFacesDetector import TinyFacesDetector
import os
from Utils import Utils
from FaceAligner import FaceAligner
import glob
import numpy as np
import cv2
import util
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


faces_out_folder = "./output/"
VideoDirectoryPath="FinalVideos/"
model_pkl="weights.pkl"
Utils.mkdir_if_not_exist(faces_out_folder)
tiny_faces_detector = TinyFacesDetector(model_pkl,use_gpu=True)
#loading mask net model
maskNet = load_model('mask_detector.model')

import numpy as np
import cv2
from random import *

## transparency value
alpha = 0.7
txt_box_height = 36
txt_box_width = 60
xoffset = 2
yoffset = 15
font_style = cv2.FONT_HERSHEY_DUPLEX
font_size = 0.4
rect_area_threshold = 1500

def overlay_bb(image, bboxes):
    # create two copies of the original image -- one for
    # the overlay and one for the final output image
    overlay = image.copy()
    output = image.copy()
    print("=======================================")
    print (len(bboxes))

    for bbox in bboxes:


        # draw a red rectangle surrounding Adrian in the image
        # along with the text "PyImageSearch" at the top-left
        # corner
        face_start_pt = (int(bbox[0]),int(bbox[1]))
        face_end_pt = (int(bbox[2]),int(bbox[3]))
        face_rect_centre_x = int((face_start_pt[0] + face_end_pt[0])/2)

        #Get values
        temp = round(uniform(97, 99.5),2)

        #cropping face using bounding box
        face = image[ int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2]) ]
        # print (face.shape)
        height, width, channels = face.shape
        face_area = height*width
        # cv2.imshow("crop", face)
        # cv2.waitKey(10)
        #For MaskNet inferencing
        face = cv2.cvtColor (face, cv2.COLOR_BGR2RGB)
        face = cv2.resize (face, (224, 224))
        face = img_to_array(face)
        face = preprocess_input (face)
        face = np.expand_dims (face, axis=0)
        preds = maskNet.predict (face)
        print(face_area)
        print(preds)
        if preds[0][0] > 0.5:
            Label= "Mask"
        elif preds[0][1] > 0.7:
            Label= "No Mask"
        else:
            Label= ""

        # Set color of overlays
        if(temp > 99.5 or Label=="No Mask"):
            rectangle_color = (30, 30, 255) #red
        else:
            rectangle_color = (30, 150, 30) #green

        #Make face rectangle overlay
        cv2.rectangle(overlay, face_start_pt, face_end_pt,
            rectangle_color, 3)

        if(face_area> rect_area_threshold):

            #TextBox below the face bb
            txt_box_start_pt = (face_rect_centre_x - int(txt_box_width/2), face_end_pt[1])
            txt_box_end_pt = (face_rect_centre_x + int(txt_box_width/2), face_end_pt[1] + txt_box_height)

            #TextBox above the face bb
        #		txt_box_start_pt = (face_rect_centre_x - txt_box_width/2, face_start_pt[1]- txt_box_height)
        #		txt_box_end_pt = (face_rect_centre_x + txt_box_width/2, face_start_pt[1] )


            cv2.rectangle(overlay, txt_box_start_pt, txt_box_end_pt,
                rectangle_color, -1)

            #Write text after overlaying to have clear font colors
            txt_pos = (txt_box_start_pt[0]+xoffset, txt_box_start_pt[1]+15)
            cv2.putText(overlay, str(temp)+" F",
                txt_pos, font_style, font_size, (255, 255, 255), 1)

            txt_pos = (txt_pos[0],txt_pos[1]+yoffset)
            cv2.putText(overlay, Label,
                txt_pos, font_style, font_size, (255, 255, 255), 1)
    
    # apply the overlay
    cv2.addWeighted(overlay, alpha, output, 1 - alpha,
        0, output)

    return output

# Process the videos
for video in glob.glob(VideoDirectoryPath+"/*.webm"):
    cap = cv2.VideoCapture(video)
    base = os.path.basename(video)
    filename = os.path.splitext(base)[0]
    print(filename)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter (str(VideoDirectoryPath)+ str(filename)+'_Output.avi', cv2.VideoWriter_fourcc ('M', 'J', 'P', 'G'), 60, (frame_width,frame_height))

    while (cap.isOpened()):
        ret, frame= cap.read()
        try:
            face_rects=tiny_faces_detector.detect(frame,nms_thresh=0.1,prob_thresh=0.5,min_conf=0.9)
            # overlay_bounding_boxes (frame, face_rects, 1)
            output = overlay_bb(frame,face_rects )

            if ret == True :
                out.write (output)
                # Display the resulting frame
                cv2.imshow ('Frame', output)
                # Press Q on keyboard to  exit
                if cv2.waitKey (5) & 0xFF == ord ('q') :
                    break

        except:
            break
    out.release()
    cap.release()

