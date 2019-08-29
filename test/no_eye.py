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

DEBUG = True
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
P3D_STOMION = numpy.float32([10.0, 0.0, -75.0]) #62
TRACKED_POINTS = (0, 4, 8, 12, 16, 17, 26, 27, 30, 33, 36, 39, 42, 45, 62)
ALL_POINTS = list(range(0,68)) #Used for debug only

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0],eye[3])
    ear = (A+B) / (2.0*C)
    return ear

def main():

    

    video_capture = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
    if(video_capture.isOpened() == False):
        print("Error: the resource is busy or unvailable")
    else:
        print("The video source has been opened correctly...")

    cv2.namedWindow('VIDEO')
    cv2.moveWindow('VIDEO', 20, 20)

    cam_w = int(video_capture.get(3))
    cam_h = int(video_capture.get(4))
    c_x = cam_w / 2
    c_y = cam_h / 2
    f_x = c_x / numpy.tan(60/2 * numpy.pi / 180)
    f_y = f_x
    camera_matrix = numpy.float32([[f_x, 0.0, c_x],
                                   [0.0, f_y, c_y],
                                   [0.0, 0.0, 1.0] ])
    print("Estimated camera matrix: \n" + str(camera_matrix) + "\n")
    camera_matrix = numpy.float32([[602.10618226,          0.0, 320.27333589],
                                   [         0.0, 603.55869786,  229.7537026],
                                   [         0.0,          0.0,          1.0] ])
    camera_distortion = numpy.float32([0.0, 0.0, 0.0, 0.0, 0.0])
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
    dlib_landmarks_file = "./etc/shape_predictor_68_face_landmarks.dat"
    if(os.path.isfile(dlib_landmarks_file)==False):
        print("The dlib landmarks file is missing! Use the following commands to download and unzip: ")
        print(">> wget dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        print(">> bzip2 -d shape_predictor_68_face_landmarks.dat.bz2")
        return
    my_detector = faceLandmarkDetection(dlib_landmarks_file)
    my_predictor = dlib.shape_predictor(dlib_landmarks_file)
    my_face_detector = dlib.get_frontal_face_detector()

    print("[INFO] loading facial landmark...")
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    
    

    #counter for counting detected eye,
    EYE_NOW = 0
    ANG_NOW = 0
    EYE_AVERAGE = 0
    ANG_AVERAGE=0
    ANG_COUNTER = 0
    EYE_COUNTER= 0

    EYE_AR_THRESH = 0.21                      #여기 파라미터 그냥 넣으시면 됩니다!

    ANG_THRESH = 0
    FACE_FRAME_COUNTER = 0
    FRAME1 = 4
    FRAME2 = 4
    FRAME3 = 3

    ANG_THRESH_LIST= numpy.zeros(15)
    ANG = False
    EYE_THRESH_LIST = numpy.zeros(5)
    EYE = False

    OPTIMIZE = True
    DETECTION = False

    while(True):
        ret, frame = video_capture.read()
        faces_array = my_face_detector(frame, 1)
        print("Total Faces: " + str(len(faces_array)))

        if(OPTIMIZE):
            if(len(faces_array) == 0):
                cv2.putText(frame, "Cannot detect face!", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            else:
                FACE_FRAME_COUNTER = 0
                ALARM_ON = False

        if(DETECTION):
            #DROWSINESS ALERT1 : Not Face Detection
            if(len(faces_array) == 0):
                FACE_FRAME_COUNTER += 1
                if(FACE_FRAME_COUNTER >= FRAME1):
                    if not ALARM_ON:
                        ALARM_ON = True
                    cv2.putText(frame, "DROWSINESS ALERT1!", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
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
            retval, rvec, tvec = cv2.solvePnP(landmarks_3D,
                                                  landmarks_2D,
                                                  camera_matrix, camera_distortion)
            axis = numpy.float32([[50,0,0],
                                      [0,50,0],
                                      [0,0,50]])
            imgpts, jac = cv2.projectPoints(axis, rvec, tvec, camera_matrix, camera_distortion)
            sellion_xy = (landmarks_2D[7][0], landmarks_2D[7][1])
            cv2.line(frame, sellion_xy, tuple(imgpts[1].ravel()), (0,255,0), 3) #GREEN
            cv2.line(frame, sellion_xy, tuple(imgpts[2].ravel()), (255,0,0), 3) #BLUE
            cv2.line(frame, sellion_xy, tuple(imgpts[0].ravel()), (0,0,255), 3) #RED
            modelpts, jac2 = cv2.projectPoints(landmarks_3D, rvec, tvec, camera_matrix, camera_distortion)
            rvec_matrix = cv2.Rodrigues(rvec)[0]
            proj_matrix = numpy.hstack((rvec_matrix, tvec))
            eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]
            pitch, yaw, roll = [math.radians(_) for _ in eulerAngles]
            roll = math.degrees(math.asin(math.sin(roll)))
            sellion_xy = (landmarks_2D[7][0], landmarks_2D[7][1])
            cv2.line(frame, sellion_xy, tuple(imgpts[1].ravel()), (0,255,0), 3) #GREEN
            cv2.line(frame, sellion_xy, tuple(imgpts[2].ravel()), (255,0,0), 3) #BLUE
            cv2.line(frame, sellion_xy, tuple(imgpts[0].ravel()), (0,0,255), 3) #RED
            cv2.putText(frame, "angle: {:.2f}".format(roll), (500,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            if(OPTIMIZE):
                if(ANG_NOW < 15):
                    ANG_THRESH_LIST[ANG_NOW]= roll
                    print("ANG_THRESH ",ANG_NOW)
                    ANG_AVERAGE = ANG_AVERAGE + roll
                    ANG_NOW =ANG_NOW+1
                elif (ANG_NOW == 15):
                    ANG_AVERAGE = ANG_AVERAGE/15
                    print(ANG_THRESH_LIST)
                    print("ANG AVERAGE IS", ANG_AVERAGE)
                    print("ANG AVERAGE COMPLETE")
                    ANG = True
                    ANG_THRESH = ANG_AVERAGE
                    cv2.putText(frame, "Angle Optimize Complete: {:.2f}".format(ANG_THRESH), (10,430),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,100), 2)
                    ANG_NOW =ANG_NOW+1
                else:
                    break

            if(DETECTION):
                #NOT FRONTAL-GAZING
                print("ANGLE:{}".format(roll))
                if(roll > 0) :
                    roll = roll - abs(ANG_THRESH)
                else:
                    roll = roll + abs(ANG_THRESH)
                print("changed ANGLE:{}".format(roll))
                if(abs(roll) >= 35 ):
                    ANG_COUNTER = ANG_COUNTER + 1
                    cv2.putText(frame, "Not frontal-gazing", (10,cam_h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
                    print("counter: {}".format(ANG_COUNTER))
                    

                    #DROWSINESS ALERT2 : KEEP (NOT FRONTAL GAZING)
                    if (ANG_COUNTER >= FRAME2):
                        if not ALARM_ON:
                            ALARM_ON = True
                        cv2.putText(frame, "DROWSINESS ALERT2!", (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                else:
                    ANG_COUNTER = 0
                    ALARM_ON = False

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = my_face_detector(gray, 0)
            for rect in rects:
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
                cv2.putText(frame, "EAR: {:.2f}".format(ear), (300,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

                if(DETECTION):
                    if ear < EYE_AR_THRESH:
                        EYE_COUNTER += 1

                        #DROWSINESS ALERT3 : NO EYE DETECTION
                        if EYE_COUNTER >= FRAME3:
                            if not ALARM_ON:
                                ALARM_ON = True
                            cv2.putText(frame, "DROWSINESS ALERT3!", (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    else:
                        EYE_COUNTER = 0
                        ALARM_ON = False
                    #cv2.putText(frame, "EAR: {:.2f}".format(ear), (450,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        cv2.imshow('VIDEO', frame)
        if (ANG == True):
            print("########################ANG OPTIMIZE COMPLETE########################")
            cv2.putText(frame, "START DETECTING!", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            print("########################START DETECTING########################")
            print("ANG_THRESH IS", ANG_THRESH)
            ANG = False
            OPTIMIZE = False
            DETECTION = True
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    #Release the camera
    video_capture.release()

if __name__ == "__main__":
    main()
