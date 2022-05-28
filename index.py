# okay so the actual bulk of the program would be done on multithreading

# { IMPORTS }

from concurrent.futures import process
import cv2
import mediapipe as mp
import pyvjoy

from pynput.keyboard import Key, Listener
import threading

from numpy import interp
from scipy.signal import savgol_filter

from cmath import cos, pi, sin
from math import atan2, degrees, radians



# { PROGRAM CRITICAL DEFINITIONS }

joystickDevice = pyvjoy.VJoyDevice(1)
captureDevice = cv2.VideoCapture(1)
captureDeviceRes = (640, 480)

initialized = True
calibrate = True

orientationInput = [] * 21
xyz = [] * 6
xyz_first = xyz
orientationOutput = [] * 2

raw_pitch = [0, ] * 10

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

def averageProcessing():
    avgX = (orientationInput[0].x + orientationInput[3].x + orientationInput[4].x + orientationInput[5].x + orientationInput[6].x + orientationInput[7].x + orientationInput[8].x + orientationInput[9].x + orientationInput[10].x + orientationInput[11].x +
            orientationInput[12].x + orientationInput[13].x + orientationInput[14].x + orientationInput[15].x + orientationInput[16].x + orientationInput[17].x + orientationInput[18].x + orientationInput[19].x + orientationInput[20].x) / 18
    avgY = (orientationInput[0].y + orientationInput[3].y + orientationInput[4].y + orientationInput[5].y + orientationInput[6].y + orientationInput[7].y + orientationInput[8].y + orientationInput[9].y + orientationInput[10].y + orientationInput[11].y +
            orientationInput[12].y + orientationInput[13].y + orientationInput[14].y + orientationInput[15].y + orientationInput[16].y + orientationInput[17].y + orientationInput[18].y + orientationInput[19].y + orientationInput[20].y) / 18
    avgZ = (orientationInput[0].z + orientationInput[3].z + orientationInput[4].z + orientationInput[5].z + orientationInput[6].z + orientationInput[7].z + orientationInput[8].z + orientationInput[9].z + orientationInput[10].z + orientationInput[11].z +
            orientationInput[12].z + orientationInput[13].z + orientationInput[14].z + orientationInput[15].z + orientationInput[16].z + orientationInput[17].z + orientationInput[18].z + orientationInput[19].z + orientationInput[20].z) / 18

    posX = round(avgX * captureDeviceRes[0])
    posY = round(avgY * captureDeviceRes[1])

    return (posX, posY, avgZ, avgX, avgY)
    # print(f'{orientationInput[20]}')


def IOROT(y, p, r):
    matrix = [
        [cos(y)*cos(p), cos(y)*sin(p)*sin(r) - sin(y) *
        cos(r), cos(y)*sin(p)*cos(r) + sin(y)*sin(r)],
        [sin(y)*cos(p), sin(y)*sin(p)*sin(r) - cos(y) *
        cos(r), sin(y)*sin(p)*cos(r) - cos(y)*sin(r)],
        [-sin(p), cos(p)*sin(r), cos(p)*cos(r)],
    ]

    realComp = [
        [], [], []
    ]

    for i in range(len(matrix)):
        for j in matrix[i]:
            realComp[i].append(j.real)

    return [
        atan2(realComp[1][0], realComp[0][0]),
        atan2(-realComp[2][0], ((realComp[2][1])
              ** 2 + (realComp[2][2])**2)**1/2),
        atan2(realComp[2][1], realComp[2][2])
    ]


def processRotation():
    global orientationOutput
    # make up a mechanism for getting out the degrees moved from position
    # print(
    #     f"delta X: {xyz[0] - xyz_first[0]}\ndelta Y: {xyz[1] - xyz_first[1]}\ndelta Z: {xyz[2] - xyz_first[2]}",  end="\r", flush=True)

    if len(raw_pitch) >= 60:
        raw_pitch[:-10] = []

    raw_pitch.append(xyz[1] - xyz_first[1])
    s_pitch = savgol_filter(raw_pitch, len(raw_pitch)-1, 1) # smooth pitch 

    i_roll = interp(xyz[0] - xyz_first[0], [-25, 25], [-1, 1])
    i_pitch = interp(s_pitch[-1], [-40, 40], [-1, 1])

    orientationOutput = [ i_roll, i_pitch ]

    print(s_pitch[-1])

    # print(orientationOutput)


def outputVjoy():
    r, p = -orientationOutput[0], orientationOutput[1]

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

# rotationProcessThread = threading.Thread(target=processRotation)
# rotationProcessThread.start()

# handUpdate = threading.Thread(target=handUpdate, args=cameraUpdate())
# handUpdate.start()

# { MAIN PROGRAM }

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

    xyz = averageProcessing()

    if calibrate:
        xyz_first = xyz
        calibrate = False

    cv2.circle(img, (xyz_first[0], xyz_first[1]), 15, (0, 255, 0), cv2.FILLED)
    cv2.circle(img, (xyz[0], xyz[1]), 15, (255, 0, 0), cv2.FILLED)
    cv2.line(img, (xyz[0], xyz[1]), (xyz_first[0], xyz_first[1]), (0, 0, 255), 2)


    processRotation()
    outputVjoy()

    cv2.imshow("Image", img)
    cv2.waitKey(1)
