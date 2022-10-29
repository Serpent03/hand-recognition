# okay so the actual bulk of the program would be done on multithreading

# { IMPORTS }

import cv2
import mediapipe as mp
import pyvjoy

import threading

import numpy as np
from scipy.signal import savgol_filter

from cmath import cos, pi, sin
from math import atan2, degrees, radians, sqrt

from kalmanfilter import KalmanFilter

# { PROGRAM CRITICAL DEFINITIONS }

kf = KalmanFilter()

joystickDevice = pyvjoy.VJoyDevice(1)
captureDevice = cv2.VideoCapture(1)
captureDeviceRes = (640, 480)

initialized = True
calibrate = True

orientationInput = [0,] * 3
xyz = [] * 3
xyz_first = xyz
orientationOutput = [0.5 , 0.5]

raw_pitch = [0, ] * 3
# s_pitch, s_roll, s_yaw = [0, ] * 3

# { MEDIAPIPE HAND TRACKING PARAMETERS }

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
mpDrawingStyles = mp.solutions.drawing_styles


# { FUNCTIONS }

def masterController(arg):
    global initialized, calibrate
    if arg == 'q':
        initialized = False
    if arg == 'c':
        calibrate = True

def rotation_matrix_from_vectors(vec1, vec2):
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    if any(v): #if not all zeros then 
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))

    else:
        return np.eye(3) #cross of all zeros only occurs on identical directions

def vectorize(orientationInp):
    vectors = [
        [orientationInp[3].x - orientationInp[0].x, orientationInp[3].y - orientationInp[0].y, orientationInp[3].z - orientationInp[0].z],
        [orientationInp[5].x - orientationInp[0].x, orientationInp[5].y - orientationInp[0].y, orientationInp[5].z - orientationInp[0].z],
        [orientationInp[17].x - orientationInp[3].x, orientationInp[17].y - orientationInp[3].y, orientationInp[17].z - orientationInp[3].z],
    ]

    return vectors

def rotFromMat():
    originalVectors = xyz_first
    transformedVectors = xyz

    rotMat1 = rotation_matrix_from_vectors(originalVectors[0], transformedVectors[0])
    rotMat2 = rotation_matrix_from_vectors(originalVectors[1], transformedVectors[1])
    rotMat3 = rotation_matrix_from_vectors(originalVectors[2], transformedVectors[2])

    x1 = np.rad2deg(atan2(rotMat1[2][1], rotMat1[2][2]))
    y1 = np.rad2deg(atan2(-rotMat1[2][0], sqrt(rotMat1[2][1] + rotMat1[2][2])))
    z1 = np.rad2deg(atan2(rotMat1[1][0], rotMat1[0][0]))

    x2 = np.rad2deg(atan2(rotMat2[2][1], rotMat2[2][2]))
    y2 = np.rad2deg(atan2(-rotMat2[2][0], sqrt(rotMat2[2][1] + rotMat2[2][2])))
    z2 = np.rad2deg(atan2(rotMat2[1][0], rotMat2[0][0]))

    x3 = np.rad2deg(atan2(rotMat3[2][1], rotMat3[2][2]))
    y3 = np.rad2deg(atan2(-rotMat3[2][0], sqrt(rotMat3[2][1] + rotMat3[2][2])))
    z3 = np.rad2deg(atan2(rotMat3[1][0], rotMat3[0][0]))

    processed = integrateRotation([x1, y1 ,z1], [x2, y2 ,z2], [x3, y3 ,z3])
    return processed

def integrateRotation(rotation1, rotation2, rotation3):
    # global s_pitch

    pitch = (rotation1[2] + rotation2[2]) / 2
    roll = -rotation3[0]

    s_pitch_roll = kf.predict(pitch, roll)
    i_pitch = np.interp(s_pitch_roll[0], [-15, 15], [0, 1])
    i_roll = np.interp(s_pitch_roll[1], [-10, 10], [0, 1])
    
    return [i_pitch, i_roll]

def outputVjoy():
    p, r = orientationOutput[0], orientationOutput[1]

    joystickDevice.set_axis(pyvjoy.HID_USAGE_X, eval(hex(int(r * 32768))))
    joystickDevice.set_axis(pyvjoy.HID_USAGE_Y, eval(hex(int(p * 32768))))

def keyUpdate():
    key = input()
    masterController(key)
    if not initialized:
        exit
    else:
        keyUpdate()


keyUpdateThread = threading.Thread(target=keyUpdate)
keyUpdateThread.start()

# { MAIN PROGRAM }

print()

while initialized:
    success, img = captureDevice.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if (results.multi_hand_landmarks):
        for handLandMarks in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(
                img,
                handLandMarks,
                mpHands.HAND_CONNECTIONS,
            )

    try:
        orientationInput = results.multi_hand_landmarks[0].landmark
    except:
        pass

    xyz = vectorize(orientationInput)

    if calibrate:
        xyz_first = xyz
        calibrate = False
    else:
        try:
            orientationOutput = rotFromMat()
        except:
            pass
        print(orientationOutput[1], "\r", end="")
    # print(xyz)

    outputVjoy()

    cv2.imshow("Image", img)
    cv2.waitKey(1)
