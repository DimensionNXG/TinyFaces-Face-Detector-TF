from TinyFacesDetector import TinyFacesDetector
import dlib
from Utils import Utils
from FaceAligner import FaceAligner
import glob
import numpy as np
import cv2
import util
import cvlib as cv
faces_out_folder = "./output/"
VideoDirectoryPath="InputVideos/"
model_pkl="weights.pkl"
Utils.mkdir_if_not_exist(faces_out_folder)
tiny_faces_detector = TinyFacesDetector(model_pkl,use_gpu=True)
font_style = cv2.FONT_HERSHEY_DUPLEX
font_size = 0.4

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
    label, confidence = cv.detect_gender (face)
    if confidence[0]> confidence[1]:
        gender= "Male"
    else:
        gender= "Female"
    print(str(np.shape(face)) +"="+ str(label))
    cv2.putText (raw_img, str(gender),
                 (_r[0], _r[1]), font_style, font_size, (255, 255, 255), 1)


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
            overlay_bounding_boxes (frame, face_rects, 1)

            if ret == True :
                # Display the resulting frame
                cv2.imshow ('Frame', frame)
                # Press Q on keyboard to  exit
                if cv2.waitKey (25) & 0xFF == ord ('q') :
                    break
            #overlay_bounding_boxes (img, face_rects , 1)
            #print(np.shape(img))
            #cv2.imshow("Frame",img)
            #print(face_rects)
            face_indx+=1
            for rect in face_rects:
                #overlay_bounding_boxes(image_path,rect,0.1)
                aligner_tight.out_dir= faces_out_folder
                aligner_tight.align_face(img1,dlib.rectangle(rect[0],rect[1],rect[2],rect[3]),str(face_indx)+'.jpg')
        except:
            break
    cap.release()
    cap1.release()


