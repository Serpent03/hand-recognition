# also a module that exports the recorded hand movements as axis signals.
# perhaps routing through vJoy.. ?
# also a possible avenue -> can we use hand motion macros to improve hand motion timings ..?

########## ########## ########## DEPRECATED ########## ########## ##########
########## ########## ########## DEPRECATED ########## ########## ##########
########## ########## ########## DEPRECATED ########## ########## ##########
########## ########## ########## DEPRECATED ########## ########## ##########

import time
# use this for the matrix transformation
from cmath import cos, pi, sin
from math import atan2, degrees, radians
from os import system

import cv2
import mediapipe as mp
import pyvjoy
from numpy import interp
# use this for the smoothening of the axis I/O
from scipy.signal import savgol_filter

# conversion from cv2 to a color image through capture camera sensor data

# camHeight = 640
# camWidth = 480

initRun = True
joystickDevice = pyvjoy.VJoyDevice(1)

targetPoints = [4, 6, 10, 14, 18]

initialX = []
initialY = []
initialZ = []
initialSlider = []

initialHandPosit = []

global currentX, currentY, currentZ, currentHandPosit
currentX = [0] * len(targetPoints)
currentY = [0] * len(targetPoints)
currentZ = [0] * len(targetPoints)
currentHandPosit = [0] * 21

raw_pitch = []
raw_roll = []
proc_Pitch = 0
proc_Roll = 0

pitchSum = 0
rollSum = 0

captureDevice = cv2.VideoCapture(1)
# captureDevice.set(3, camWidth)
# captureDevice.set(4, camHeight)

mpHands = mp.solutions.hands
# the class function already has the default values needed
hands = mpHands.Hands(min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def appendX(drawRes):
    currentX[0] = (initialX[0] - drawRes.multi_hand_landmarks[0].landmark[4].x)
    currentX[1] = (initialX[1] - drawRes.multi_hand_landmarks[0].landmark[6].x)
    currentX[2] = (initialX[2] - drawRes.multi_hand_landmarks[0].landmark[10].x)
    currentX[3] = (initialX[3] - drawRes.multi_hand_landmarks[0].landmark[14].x)
    currentX[4] = (initialX[4] - drawRes.multi_hand_landmarks[0].landmark[18].x)


def appendY(drawRes):
    currentY[0] = (initialY[0] - drawRes.multi_hand_landmarks[0].landmark[4].y)
    currentY[1] = (initialY[1] - drawRes.multi_hand_landmarks[0].landmark[6].y)
    currentY[2] = (initialY[2] - drawRes.multi_hand_landmarks[0].landmark[10].y)
    currentY[3] = (initialY[3] - drawRes.multi_hand_landmarks[0].landmark[14].y)
    currentY[4] = (initialY[4] - drawRes.multi_hand_landmarks[0].landmark[18].y)


def appendZ(drawRes):
    currentZ[0] = (initialZ[0] - drawRes.multi_hand_landmarks[0].landmark[4].z)
    currentZ[1] = (initialZ[1] - drawRes.multi_hand_landmarks[0].landmark[6].z)
    currentZ[2] = (initialZ[2] - drawRes.multi_hand_landmarks[0].landmark[10].z)
    currentZ[3] = (initialZ[3] - drawRes.multi_hand_landmarks[0].landmark[14].z)
    currentZ[4] = (initialZ[4] - drawRes.multi_hand_landmarks[0].landmark[18].z)

def appendHand(drawRes):
    for i in range(21):
        currentHandPosit.append(drawRes.multi_hand_landmarks[0].landmark[i])


def drawImage(drawRes):
    global initRun

    if (results.multi_hand_landmarks):    
        if initRun:
            for i in range(21):
                initialHandPosit.append(drawRes.multi_hand_landmarks[0].landmark[i])
            initRun = False
            for i in targetPoints:
                initialX.append(drawRes.multi_hand_landmarks[0].landmark[i].x)
                initialY.append(drawRes.multi_hand_landmarks[0].landmark[i].y)
                initialZ.append(drawRes.multi_hand_landmarks[0].landmark[i].z)
        else:
            appendX(drawRes)
            appendY(drawRes)
            appendZ(drawRes)
            appendHand(drawRes)


def calculateRotation(ix, iy, iz):
    # start using matrix here...

    # in current config:

    # ROLL -> ± Z, two extremeties
    # PITCH -> ± X
    # YAW -> ± Z, all points

    # let's see -> if we want to smoothen it out, the simplest way is to
    # take an average out of some samples. the program is fast enough for only 2-3 ..

    global raw_pitch, raw_roll, proc_Pitch, proc_Roll, pitchSum, rollSum

    raw_pitch.append(round(currentY[2], 7))
    raw_roll.append(round((currentZ[0])*2, 7))

    if len(raw_pitch) >= 2:
        for dps in raw_pitch:  # datapoints
            pitchSum += dps
        proc_Pitch = 0
        proc_Pitch = pitchSum/2
        raw_pitch = []

    if len(raw_roll) >= 2:
        for dps in raw_roll:  # datapoints
            rollSum += dps
        proc_Roll = 0
        proc_Roll = rollSum/2
        raw_roll = []

    # pitch = round(iy[2] - currentY[2], 7) # -0.1 -> 0.1
    # roll = round((iz[2] - currentZ[0])*2, 7) # -0.05 -> 0.05

    i_pitch = interp(proc_Pitch, [-0.1, 0.1], [-1, 1])
    i_roll = interp(proc_Roll, [-0.01, 0.08], [-1, 1])

    # print(pitch, proc_Pitch)

    pitchSum = 0
    rollSum = 0
    return [i_pitch, -i_roll]


def outputVjoy(rotParams):
    p, r = rotParams[0], rotParams[1]

    # the axis range is from 0x0000 to 0x8000, which is 0 to 32768
    # we simply put a multiplier to 32768, convert it to base10, convert
    # to a hex string, and then eval() to convert it into base16

    joystickDevice.set_axis(pyvjoy.HID_USAGE_Y, eval(hex(int(p * 32768))))
    joystickDevice.set_axis(pyvjoy.HID_USAGE_X, eval(hex(int(r * 32768))))

def MATROT(ix, iy, iz):
    pass



while True:
    success, img = captureDevice.read()
    # img.flags.writeable = False
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    drawImage(results)
    rot = calculateRotation(initialX, initialY, initialZ)
    testRot = MATROT(initialX, initialY, initialZ)
    outputVjoy(rot)

    if (results.multi_hand_landmarks):
        for handLandMarks in results.multi_hand_landmarks:
            # finger joint landmark 0 through 20
            # for rotational data, it looks like we will have to compare the
            # positions inside targetpoints against initial positions, and then we apply some
            # clever trigonometry to get the angle traversed <- this step is not put in yet
            mpDraw.draw_landmarks(
                img,
                handLandMarks,
                mpHands.HAND_CONNECTIONS,
            )
            print(currentHandPosit)

        # print(type(initialHandPosit[0]))
        # break

    cv2.imshow("Image", img)
    cv2.waitKey(1)
