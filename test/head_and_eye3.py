#!/usr/bin/env python

##
# Massimiliano Patacchiola, Plymouth University 2016
#
# This is an example of head pose estimation with solvePnP.
# It uses the dlib library and openCV
#

import numpy 
import math
import cv2
import imutils
import time
import dlib
import sys
import os
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
from deepgaze.face_landmark_detection import faceLandmarkDetection

def eye_aspect_ratio(eye):
    # compute 눈의 세로길이
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # 눈의 가로길이
    C = dist.euclidean(eye[0],eye[3])

    # eye aspect ratio계산
    ear = (A+B) / (2.0*C)

    return ear

#If True enables the verbose mode
DEBUG = True


#Antropometric constant values of the human head. 
#Found on wikipedia and on:
# "Head-and-Face Anthropometric Survey of U.S. Respirator Users"
#
#X-Y-Z with X pointing forward and Y on the left.
#The X-Y-Z coordinates used are like the standard
# coordinates of ROS (robotic operative system)
P3D_RIGHT_SIDE = numpy.float32([-100.0, -77.5, -5.0]) #0
P3D_GONION_RIGHT = numpy.float32([-110.0, -77.5, -85.0]) #4
P3D_MENTON = numpy.float32([0.0, 0.0, -122.7]) #8
P3D_GONION_LEFT = numpy.float32([-110.0, 77.5, -85.0]) #12
P3D_LEFT_SIDE = numpy.float32([-100.0, 77.5, -5.0]) #16
P3D_FRONTAL_BREADTH_RIGHT = numpy.float32([-20.0, -56.1, 10.0]) #17
P3D_FRONTAL_BREADTH_LEFT = numpy.float32([-20.0, 56.1, 10.0]) #26
P3D_SELLION = numpy.float32([0.0, 0.0, 0.0]) #27
P3D_NOSE = numpy.float32([21.1, 0.0, -48.0]) #30
P3D_SUB_NOSE = numpy.float32([5.0, 0.0, -52.0]) #33
P3D_RIGHT_EYE = numpy.float32([-20.0, -65.5,-5.0]) #36
P3D_RIGHT_TEAR = numpy.float32([-10.0, -40.5,-5.0]) #39
P3D_LEFT_TEAR = numpy.float32([-10.0, 40.5,-5.0]) #42
P3D_LEFT_EYE = numpy.float32([-20.0, 65.5,-5.0]) #45
#P3D_LIP_RIGHT = numpy.float32([-20.0, 65.5,-5.0]) #48
#P3D_LIP_LEFT = numpy.float32([-20.0, 65.5,-5.0]) #54
P3D_STOMION = numpy.float32([10.0, 0.0, -75.0]) #62

#The points to track
#These points are the ones used by PnP
# to estimate the 3D pose of the face
TRACKED_POINTS = (0, 4, 8, 12, 16, 17, 26, 27, 30, 33, 36, 39, 42, 45, 62)
ALL_POINTS = list(range(0,68)) #Used for debug only


def main():
    
    #alarm for drowsiness
    ALARM_ON = False

    #Defining the video capture object
    #video_image = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
    video_capture = cv2.VideoCapture("C:/Users/YJ/Desktop/학교생활/UROP 연구/190806 낮-20190816T135703Z-001/190806 낮/맨얼굴_2.mp4")
    _,video_image = video_capture.read()
    video_image = cv2.resize(video_image, dsize=(640,480), interpolation=cv2.INTER_AREA)
    if(video_image.isOpened() == False):
        print("Error: the resource is busy or unvailable")
    else:
        print("The video source has been opened correctly...")

    #Create the main window and move it
    cv2.namedWindow('Video')
    cv2.moveWindow('Video', 20, 20)

    #Obtaining the CAM dimension
    cam_w = int(video_image.get(3))
    cam_h = int(video_image.get(4))

    #Defining the camera matrix.
    #To have better result it is necessary to find the focal
    # lenght of the camera초점거리를 알면 좋음.
    # fx/fy are the focal lengths (in pixels) 
    # and cx/cy are the optical centres. These values can be obtained 
    # roughly by approximation, for example in a 640x480 camera:
    # cx = 640/2 = 320
    # cy = 480/2 = 240
    # fx = fy = cx/tan(60/2 * pi / 180) = 554.26
    c_x = cam_w / 2
    c_y = cam_h / 2
    f_x = c_x / numpy.tan(60/2 * numpy.pi / 180)
    f_y = f_x

    #Estimated camera matrix values.
    camera_matrix = numpy.float32([[f_x, 0.0, c_x],
                                   [0.0, f_y, c_y], 
                                   [0.0, 0.0, 1.0] ])

    print("Estimated camera matrix: \n" + str(camera_matrix) + "\n")

    #These are the camera matrix values estimated on my webcam with
    # the calibration code (see: src/calibration):
    camera_matrix = numpy.float32([[602.10618226,          0.0, 320.27333589],
                                   [         0.0, 603.55869786,  229.7537026], 
                                   [         0.0,          0.0,          1.0] ])

    #Distortion coefficients
    camera_distortion = numpy.float32([0.0, 0.0, 0.0, 0.0, 0.0])

    #Distortion coefficients estimated by calibration
    #camera_distortion = numpy.float32([ 0.06232237, -0.41559805,  0.00125389, -0.00402566,  0.04879263])


    #This matrix contains the 3D points of the
    # 11 landmarks we want to find. It has been
    # obtained from antrophometric measurement
    # on the human head.
    landmarks_3D = numpy.float32([P3D_RIGHT_SIDE,
                                  P3D_GONION_RIGHT,
                                  P3D_MENTON,
                                  P3D_GONION_LEFT,
                                  P3D_LEFT_SIDE,
                                  P3D_FRONTAL_BREADTH_RIGHT,
                                  P3D_FRONTAL_BREADTH_LEFT,
                                  P3D_SELLION,
                                  P3D_NOSE,
                                  P3D_SUB_NOSE,
                                  P3D_RIGHT_EYE,
                                  P3D_RIGHT_TEAR,
                                  P3D_LEFT_TEAR,
                                  P3D_LEFT_EYE,
                                  P3D_STOMION])

   #Declaring the two classifiers
    #my_cascade = haarCascade("../etc/haarcascade_frontalface_alt.xml", "../etc/haarcascade_profileface.xml")
    dlib_landmarks_file = "./etc/shape_predictor_68_face_landmarks.dat"
    if(os.path.isfile(dlib_landmarks_file)==False): 
        print("The dlib landmarks file is missing! Use the following commands to download and unzip: ")
        print(">> wget dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        print(">> bzip2 -d shape_predictor_68_face_landmarks.dat.bz2")
        return
    my_detector = faceLandmarkDetection(dlib_landmarks_file)
    my_predictor = dlib.shape_predictor(dlib_landmarks_file)
    my_face_detector = dlib.get_frontal_face_detector()



    #settings for detection
    
    #detect blink
    EYE_AR_THRESH = 0.15
    # number of consecutive frames. 이거넘으면 알람
    ANG_THRESH = 55.0

    #counter for counting undetected face
    FACE_FRAME_COUNTER = 0
    
    #counter for counting not frontal-gazing frames
    ANG_COUNTER = 0

    #counter for counting closed eyes
    EYE_COUNTER= 0

    #number of consequtive frames for face detection 
    FRAME1 = 3

    #number of consequtive frames for frontal gaze detection 
    FRAME2 = 3
    
    #number of consequtive frames for eye detection 
    FRAME3 = 3

    

    

    print("[INFO] loading facial landmark predictor...")
    

    # grab the indexes of the facial landmarks for the left and
    # right eye, respectively
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


    while(True):

        # Capture frame-by-frame
        ret, frame = video_image.read()
        #gray = cv2.cvtColor(frame[roi_y1:roi_y2, roi_x1:roi_x2], cv2.COLOR_BGR2GRAY)
            
        faces_array = my_face_detector(frame, 1)  
        print("Total Faces: " + str(len(faces_array)))
        

        if(faces_array == None):
            FACE_FRAME_COUNTER += 1

            if(FACE_FRAME_COUNTER >= FRAME1):
                if not ALARM_ON:
                        ALARM_ON = True
                cv2.putText(frame, "DROWSINESS ALERT1!", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                  
        else:
            FACE_FRAME_COUNTER = 0
            ALARM_ON = False

            
        
        for i, pos in enumerate(faces_array):

            face_x1 = pos.left()
            face_y1 = pos.top()
            face_x2 = pos.right()
            face_y2 = pos.bottom()
            text_x1 = face_x1
            text_y1 = face_y1 - 3

            cv2.putText(frame, "FACE " + str(i+1), (text_x1,text_y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1);
            cv2.rectangle(frame, 
                         (face_x1, face_y1), 
                         (face_x2, face_y2), 
                         (0, 255, 0), 
                          2)            

            landmarks_2D = my_detector.returnLandmarks(frame, face_x1, face_y1, face_x2, face_y2, points_to_return=TRACKED_POINTS)


            for point in landmarks_2D:
                cv2.circle(frame,( point[0], point[1] ), 2, (0,0,255), -1)


            #Applying the PnP solver to find the 3D pose
            # of the head from the 2D position of the
            # landmarks.
            #retval - bool
            #rvec - Output rotation vector that, together with tvec, brings 
            # points from the model coordinate system to the camera coordinate system.
            #tvec - Output translation vector.
            retval, rvec, tvec = cv2.solvePnP(landmarks_3D, 
                                                  landmarks_2D, 
                                                  camera_matrix, camera_distortion)

            #Now we project the 3D points into the image plane
            #Creating a 3-axis to be used as reference in the image.
            axis = numpy.float32([[50,0,0], 
                                      [0,50,0], 
                                      [0,0,50]])
            imgpts, jac = cv2.projectPoints(axis, rvec, tvec, camera_matrix, camera_distortion)

            #Drawing the three axis on the image frame.
            #The opencv colors are defined as BGR colors such as: 
            # (a, b, c) >> Blue = a, Green = b and Red = c
            #Our axis/color convention is X=R, Y=G, Z=B
            sellion_xy = (landmarks_2D[7][0], landmarks_2D[7][1])
            cv2.line(frame, sellion_xy, tuple(imgpts[1].ravel()), (0,255,0), 3) #GREEN
            cv2.line(frame, sellion_xy, tuple(imgpts[2].ravel()), (255,0,0), 3) #BLUE
            cv2.line(frame, sellion_xy, tuple(imgpts[0].ravel()), (0,0,255), 3) #RED
       

            #Compute rotate_degree
            modelpts, jac2 = cv2.projectPoints(landmarks_3D, rvec, tvec, camera_matrix, camera_distortion)
            rvec_matrix = cv2.Rodrigues(rvec)[0]
            
            proj_matrix = numpy.hstack((rvec_matrix, tvec))
            eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]

            pitch, yaw, roll = [math.radians(_) for _ in eulerAngles]

            roll = math.degrees(math.asin(math.sin(roll)))
            roll = roll - 20
            print(str(roll))
            
            #Drawing the three axis on the image frame.
            #The opencv colors are defined as BGR colors such as: 
            # (a, b, c) >> Blue = a, Green = b and Red = c
            #Our axis/color convention is X=R, Y=G, Z=B
            sellion_xy = (landmarks_2D[7][0], landmarks_2D[7][1])
            cv2.line(frame, sellion_xy, tuple(imgpts[1].ravel()), (0,255,0), 3) #GREEN
            cv2.line(frame, sellion_xy, tuple(imgpts[2].ravel()), (255,0,0), 3) #BLUE
            cv2.line(frame, sellion_xy, tuple(imgpts[0].ravel()), (0,0,255), 3) #RED

            cv2.putText(frame, "angle: {:.2f}".format(roll), (300,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            if(abs(roll) >= ANG_THRESH ):
                ANG_COUNTER = ANG_COUNTER + 1
                cv2.putText(frame, "Not frontal-gazing", (10,cam_h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
                
                print("counter: {}".format(ANG_COUNTER))
                
                if (ANG_COUNTER >= FRAME2):
                    if not ALARM_ON:
                        ALARM_ON = True
                        
                    cv2.putText(frame, "DROWSINESS ALERT2!", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            else:
                ANG_COUNTER = 0
                ALARM_ON = False

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rects = my_face_detector(gray, 0)

                
                
                


                for rect in rects:
                    # determine the facial landmarks for the face region, then
                    # convert the facial landmark (x, y)-coordinates to a NumPy
                    # array
                    shape = my_predictor(gray, rect)
                    shape = face_utils.shape_to_np(shape)

                    leftEye = shape[lStart:lEnd]
                    rightEye = shape[rStart:rEnd]
                    leftEAR = eye_aspect_ratio(leftEye)
                    rightEAR = eye_aspect_ratio(rightEye)

                    ear = (leftEAR + rightEAR) / 2.0

                    leftEyeHull = cv2.convexHull(leftEye)
                    rightEyeHull = cv2.convexHull(rightEye)
                    cv2.drawContours(frame,[leftEyeHull], -1,(0,255,0),1)
                    cv2.drawContours(frame,[rightEyeHull], -1,(0,255,0),1)

                    if ear < EYE_AR_THRESH:
                        EYE_COUNTER += 1

                        if EYE_COUNTER >= FRAME3:

                            if not ALARM_ON:
                                ALARM_ON = True

                            cv2.putText(frame, "DROWSINESS ALERT3!", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    else:
                        EYE_COUNTER = 0
                        ALARM_ON = False

                    cv2.putText(frame, "EAR: {:.2f}".format(ear), (300,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)


   

                

            

        #Showing the frame and waiting
        # for the exit command
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
   
    #Release the camera
    video_image.release()
    print("Bye...")



if __name__ == "__main__":
    main()
