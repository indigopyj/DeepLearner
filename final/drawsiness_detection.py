'''
    This program detects the drowsiness of a driver in real time with using OpenCV Python.

    Copyright @ DeepLearner From Ewha Womans Univ.

    Ref from Deepgaze https://github.com/mpatacchiola/deepgaze,

             PyImageSearch https://www.pyimagesearch.com/2017/05/08/drowsiness-detection-opencv/
            
'''





import winsound as ws
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

# X-Y-Z with X pointing forward and Y on the left.
# The X-Y-Z coordinates used are like the standard
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
P3D_STOMION = numpy.float32([10.0, 0.0, -75.0]) #62

# The points to track
# These points are the ones used by PnP to estimate the 3D pose of the face
TRACKED_POINTS = (0, 4, 8, 12, 16, 17, 26, 27, 30, 33, 36, 39, 42, 45, 62)
ALL_POINTS = list(range(0,68)) #Used for debug only


def beepsound():
    freq = 2000    # range : 37 ~ 32767
    dur = 500     # ms
    ws.Beep(freq, dur) # winsound.Beep(frequency, duration)

#calculate ROI of eyes
def eye_aspect_ratio(eye):
    # compute the vertical length of eye
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the horizontal length of eye
    C = dist.euclidean(eye[0],eye[3])

    # compute eye aspect ratio
    ear = (A+B) / (2.0*C)
    return ear

def main():
    # Open the video file
    video_capture = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
    if(video_capture.isOpened() == False):
        print("Error: the resource is busy or unvailable")
    else:
        print("The video source has been opened correctly...")

    # Create the main window and move it
    cv2.namedWindow('VIDEO')
    cv2.moveWindow('VIDEO', 20, 20)
    
    # Obtaining the CAM dimension
    cam_w = int(video_capture.get(3))
    cam_h = int(video_capture.get(4))

    # Defining the camera matrix.
    # To have better result it is necessary to find the focal
    # lenght of the camera. fx/fy are the focal lengths (in pixels) 
    # and cx/cy are the optical centres.
    c_x = cam_w / 2
    c_y = cam_h / 2
    f_x = c_x / numpy.tan(60/2 * numpy.pi / 180)
    f_y = f_x

    camera_matrix = numpy.float32([[f_x, 0.0, c_x],
                                   [0.0, f_y, c_y],
                                   [0.0, 0.0, 1.0] ])
    print("Estimated camera matrix: \n" + str(camera_matrix) + "\n")
    
    camera_distortion = numpy.float32([0.0, 0.0, 0.0, 0.0, 0.0])

    # load 68 face landmarks using dlib and detect the face
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

    # Declaring the classifier
    dlib_landmarks_file = "./etc/shape_predictor_68_face_landmarks.dat"
    
    if(os.path.isfile(dlib_landmarks_file)==False):
        print("The dlib landmarks file is missing! Use the following commands to download and unzip: ")
        print(">> wget dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        print(">> bzip2 -d shape_predictor_68_face_landmarks.dat.bz2")
        return
    my_detector = faceLandmarkDetection(dlib_landmarks_file)
    my_predictor = dlib.shape_predictor(dlib_landmarks_file)
    my_face_detector = dlib.get_frontal_face_detector()

    # grab the indexes of the facial landmarks for the left and right eye, respectively
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]



    # factors for detecting drowsiness
    
    ANG_NOW = 0             # counter for optimizing the optimized angle of driver
    ROLL_AVERAGE=0          # result of an average of optimized roll angle
    PITCH_AVERAGE=0         # result of an average of optimized pitch angle
    ANG_OPTIMIZE = 0        # result of an optimized angle
    
    ANG_COUNTER = 0         # counter for counting detected angle
    EYE_COUNTER= 0          # counter for counting detected eye
    FACE_FRAME_COUNTER = 0  # counter for counting non-face-detected frame

    EYE_AR_THRESH = 0.21    # a threshold for detecting blink                      
    ANG_THRESH = 7          # a threshold for detecting bowing head

    
    FRAME1 = 4              # number of consecutive frames of non-face-detection
    FRAME2 = 4              # number of consecutive frames of detected angle
    FRAME3 = 3              # number of consecutive frames of detected eye

    # variables needed for optimization
    ROLL_THRESH_LIST= numpy.zeros(15)
    PITCH_THRESH_LIST = numpy.zeros(15)
    ANG = False
    OPTIMIZE = True
    OPTIMIZE_AGAIN = True
    DETECTION = False

    while(True):
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        faces_array = my_face_detector(frame, 1)    # get a face array from face detection
        print("Total Faces: " + str(len(faces_array)))

        
        
        if(OPTIMIZE):
            if(len(faces_array) == 0):  # if face detection is failed
                cv2.putText(frame, "Cannot detect face!", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            else:
                FACE_FRAME_COUNTER = 0
                ALARM_ON = False

        # Detection : Detect drowsiness when a particular situation lasts for a number of frames
        if(DETECTION):
            #DROWSINESS ALERT1 : Non-face-detection including the case when the driver bends one's head too much
            if(len(faces_array) == 0):
                FACE_FRAME_COUNTER += 1
                if(FACE_FRAME_COUNTER >= FRAME1):
                    if not ALARM_ON:
                        ALARM_ON = True
                    cv2.putText(frame, "DROWSINESS ALERT1!", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    print("**DROWSINESS ALERT1!**\n")
                    beepsound()
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

            # Applying the PnP solver to find the 3D pose of the head from the 2D position of the landmarks.
            # retval - bool
            # rvec - Output rotation vector that, together with tvec, brings points from the model coordinate system to the camera coordinate system.
            # tvec - Output translation vector.
            retval, rvec, tvec = cv2.solvePnP(landmarks_3D,
                                                  landmarks_2D,
                                                  camera_matrix, camera_distortion)
            # Now we project the 3D points into the image plane
            # Creating a 3-axis to be used as reference in the image.
            axis = numpy.float32([[50,0,0],
                                      [0,50,0],
                                      [0,0,50]])
            imgpts, jac = cv2.projectPoints(axis, rvec, tvec, camera_matrix, camera_distortion)

            # Drawing the three axis on the image frame.
            # The opencv colors are defined as BGR colors such as: 
            # (a, b, c) >> Blue = a, Green = b and Red = c
            #  Our axis/color convention is X=R, Y=G, Z=B
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
            pitch = math.degrees(math.asin(math.sin(pitch)))
            yaw = math.degrees(math.asin(math.sin(yaw)))
            #print(pitch, yaw, roll)
            cv2.putText(frame, "Roll: {:.2f}".format(roll), (500,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.putText(frame, "Pitch: {:.2f}".format(pitch), (500,80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)


            # Optimization : calculate the average angle when the driver is looking at the front
            if (OPTIMIZE):
                if(ANG_NOW < 15): # get roll,and pitch during 15 frames
                    ROLL_THRESH_LIST[ANG_NOW]= roll
                    PITCH_THRESH_LIST[ANG_NOW] = pitch
                    print("ANG_NOW ",ANG_NOW)
                    ANG_NOW =ANG_NOW+1
                elif (ANG_NOW == 15):
                    print(ROLL_THRESH_LIST)
                    print(PITCH_THRESH_LIST)
                    listR = []
                    listPp = []
                    listPn= []
                    listP = []
                    pos = 0
                    neg = 0

                    # remove the outliers
                    for i in ROLL_THRESH_LIST:
                        if(i >=-90 and i<=-70):
                            listR.append(i)
                    for i in PITCH_THRESH_LIST:
                        #if most of pitch values are positive, then we will use only positive values for computing the avg.
                        if(i >= 0):
                            pos = pos+1
                            listPp.append(i)
                        #if most of pitch values are negative, then we will use only negative values for computing the avg.    
                        else:
                            neg = neg + 1
                            listPn.append(i)
                    if(pos > neg):
                        listP = listPp
                    else:
                        listP = listPn
                        
                    print(listR)
                    print(listP)
                    if not listR or not listP: #if either of lists is null, optimize again
                        beepsound()
                        cv2.putText(frame, "Angle Optimize Fail", (10,430), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 255), 2)
                        ANG_NOW = 0
                        
                    else:
                        ROLL_AVERAGE = numpy.mean(listR)
                        PITCH_AVERAGE = numpy.mean(listP)
                        print("ROLL AVERAGE IS", ROLL_AVERAGE)
                        print("PITCH AVERAGE IS", PITCH_AVERAGE)
                        print("ANG AVERAGE COMPLETE")
                        beepsound()
                        beepsound()
                        ANG = True
                        ROLL_OPTIMIZE = ROLL_AVERAGE
                        PITCH_OPTIMIZE = PITCH_AVERAGE
                        cv2.putText(frame, "Angle Optimize Complete", (10,430),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,100), 2)
                        ANG_NOW =ANG_NOW+1
                else:
                    break


            if(DETECTION):
                print("ROLL ANGLE:{}   PITCH ANGLE : {}".format(roll, pitch))
                # compute the distance between average and detected roll regardless of sign
                if(roll > 0) :
                    roll = roll - abs(ROLL_OPTIMIZE)
                else:
                    roll = roll + abs(ROLL_OPTIMIZE)
                # compute the distance between average and detected pitch
                pitch = PITCH_AVERAGE - pitch
                print("changed ROLL:{}".format(roll))
                print("changed PITCH:{}".format(pitch))

                # detect NOT FRONTAL-GAZING
                if(abs(roll) <= 7 and abs(pitch) >= 10 ):
                    ANG_COUNTER = ANG_COUNTER + 1
                    cv2.putText(frame, "Not frontal-gazing", (10,cam_h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
                    print("counter: {}".format(ANG_COUNTER))
                    print("Non frontal-gazing!\n");


                    #DROWSINESS ALERT2 : KEEP (NOT FRONTAL GAZING)
                    if (ANG_COUNTER >= FRAME2):
                        if not ALARM_ON:
                            ALARM_ON = True
                        cv2.putText(frame, "DROWSINESS ALERT2!", (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                        beepsound()
                        print("**DROWSINESS ALERT2!**\n");
                else:
                    ANG_COUNTER = 0
                    ALARM_ON = False

                    

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = my_face_detector(gray, 0)
            for rect in rects:
                # determine the facial landmarks for the face region, then convert
                # the facial landmark (x, y)-coordinates to a NumPy array
                shape = my_predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                
                # compute eye aspect ratio 
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                ear = (leftEAR + rightEAR) / 2.0
                
                # draw line along the eyes
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(frame,[leftEyeHull], -1,(0,255,0),1)
                cv2.drawContours(frame,[rightEyeHull], -1,(0,255,0),1)
                cv2.putText(frame, "EAR: {:.2f}".format(ear), (300,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

                if(DETECTION):
                    if ear < EYE_AR_THRESH:
                        EYE_COUNTER += 1
                        # DROWSINESS ALERT3 : closed eye detection
                        if EYE_COUNTER >= FRAME3:
                            if not ALARM_ON:
                                ALARM_ON = True
                            cv2.putText(frame, "DROWSINESS ALERT3!", (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                            print("**DROWSINESS ALERT3!**\n")
                            beepsound()
                    else:
                        EYE_COUNTER = 0
                        ALARM_ON = False

        cv2.imshow('VIDEO', frame)
        if (ANG == True): # Optimization complete
            print("########################ANG OPTIMIZE COMPLETE########################")
            cv2.putText(frame, "START DETECTING!", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            print("########################START DETECTING########################")
            print("ROLL_OPTIMIZE IS", ROLL_OPTIMIZE)
            print("PITCH_OPTIMIZE IS", PITCH_OPTIMIZE)
            ANG = False
            OPTIMIZE = False
            DETECTION = True
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    #Release the camera
    video_capture.release()



if __name__ == "__main__":
    main()
