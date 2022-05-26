# okay so the actual bulk of the program would be done on multithreading

# { IMPORTS }

from ast import arguments
import cv2
import mediapipe as mp
import pyvjoy

from pynput.keyboard import Key, Listener
import threading

from numpy import interp

# { PROGRAM CRITICAL DEFINITIONS }

joystickDevice = pyvjoy.VJoyDevice(1)
captureDevice = cv2.VideoCapture(1)

initialized = True

# { MEDIAPIPE HAND TRACKING PARAMETERS }

mpHands = mp.solutions.hands
hands = mpHands.Hands(min_detection_confidence=0.6)
mpDraw = mp.solutions.drawing_utils
mpDrawingStyles = mp.solutions.drawing_styles


# { FUNCTIONS }

def masterController(arg):
    global initialized
    if arg == 'q':
        initialized = False


def stopProgram():
    global initialized
    initialized = False


def cameraUpdate():
    global success, imgRGB, img

    success, img = captureDevice.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

    return hands.process(imgRGB)


def handUpdate(r):
    if (r.multi_hand_landmarks):
        for handLandMarks in r.multi_hand_landmarks:
            mpDraw.draw_landmarks(
                img,
                handLandMarks,
                mpHands.HAND_CONNECTIONS
            )


def keyUpdate():
    key = input()
    masterController(key)


keyUpdate = threading.Thread(target=keyUpdate)
keyUpdate.start()

# handUpdate = threading.Thread(target=handUpdate, args=cameraUpdate())
# handUpdate.start()

# { MAIN PROGRAM }

while initialized:

    results = cameraUpdate()

    if (results.multi_hand_landmarks):
        for handLandMarks in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(
                img,
                handLandMarks,
                mpHands.HAND_CONNECTIONS
            )
