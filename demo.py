from TinyFacesDetector import TinyFacesDetector
import dlib
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
VideoDirectoryPath="InputVideos1/"
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
txt_box_height = 40
txt_box_width = 70
xoffset = 2
yoffset = 15
font_style = cv2.FONT_HERSHEY_DUPLEX
font_size = 0.4
rect_area_threshold = 2000

def overlay_bb(image, bboxes):
    # create two copies of the original image -- one for
    # the overlay and one for the final output image
    overlay = image.copy()
    output = image.copy()
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
        if preds[0][0] > preds[0][1]:
            Label= "Mask"
        else:
            Label= "No Mask"

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

            #TextBox below the face bb
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

#function to overlay bounding boxes
def overlay_bounding_boxes(raw_img, refined_bboxes, lw):
  # Overlay bounding boxes on an image with the color based on the confidence.
  for r in refined_bboxes:
    #_score = round(r[4])
    _score = 0.5
    cm_idx = int(np.ceil(_score * 255))
    rect_color = [int(np.ceil(x * 255)) for x in util.cm_data[cm_idx]]  # parula
    _lw = lw
    if lw == 0:  # line width of each bounding box is adaptively determined.
      bw, bh = r[2] - r[0] + 1, r[3] - r[0] + 1
      _lw = 1 if min(bw, bh) <= 20 else max(2, min(3, min(bh / 20, bw / 20)))
      _lw = int(np.ceil(_lw * _score))
    _r = [int(x) for x in r[:4]]
    cv2.rectangle(raw_img, (_r[0], _r[1]), (_r[2], _r[3]), rect_color, _lw)
    #cropping face using bounding box
    face = raw_img[r[1]:r[3], r[0]:r[2]]
    #For MaskNet inferencing
    face = cv2.cvtColor (face, cv2.COLOR_BGR2RGB)
    face = cv2.resize (face, (224, 224))
    face = img_to_array(face)
    face = preprocess_input (face)
    face = np.expand_dims (face, axis=0)
    preds = maskNet.predict (face)
    if preds[0][0] > preds[0][1]:
        Label= "Mask"
    else:
        Label= "NonMask"
    print(Label)
    cv2.putText (raw_img, str (Label),
                 (_r[0], _r[1]), font_style, font_size, (255, 255, 255), 1)

    #print((preds)[0][1])

    #resizing with border:
    """desired_size = 112
    old_size = face.shape[:2]  # old_size is in (height, width) format
    ratio = float (desired_size) / max (old_size)
    new_size = tuple ([int (x * ratio) for x in old_size])

    # new_size should be in (width, height) format

    im = cv2.resize (im, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder (im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                 value=color)
    print(np.shape(new_im))"""
#init the face landmarks detector
predictor_5_face_landmarks = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")

# tight face aligner : padding = 0.2
aligner_tight = FaceAligner(face_size=112, face_padding=0.2, predictor_5_face_landmarks=predictor_5_face_landmarks)
face_indx=0
for video in glob.glob(VideoDirectoryPath+"/*.webm"):
    cap = cv2.VideoCapture(video)
    #Creating 2 video capture object because align face is writing image with bounding box.
    cap1 = cv2.VideoCapture(video)
    #ret, img = cap.read()
    while (cap.isOpened()):
        ret, frame= cap.read()
        try:
            #next line is just for align face function.
            ret1,img1 = cap1.read()
            face_rects=tiny_faces_detector.detect(frame,nms_thresh=0.1,prob_thresh=0.5,min_conf=0.9)
            # overlay_bounding_boxes (frame, face_rects, 1)
            output = overlay_bb(frame,face_rects )

            if ret == True :
                # Display the resulting frame
                cv2.imshow ('Frame', output)
                # Press Q on keyboard to  exit
                if cv2.waitKey (25) & 0xFF == ord ('q') :
                    break
            #overlay_bounding_boxes (img, face_rects , 1)
            #print(np.shape(img))
            #cv2.imshow("Frame",img)
            #print(face_rects)
            face_indx+=1
            # for rect in face_rects:
            #     #overlay_bounding_boxes(image_path,rect,0.1)
            #     aligner_tight.out_dir= faces_out_folder
            #     aligner_tight.align_face(img1,dlib.rectangle(rect[0],rect[1],rect[2],rect[3]),str(face_indx)+'.jpg')
        except:
            break
    cap.release()
    cap1.release()

