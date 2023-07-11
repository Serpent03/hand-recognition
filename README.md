# Hand-Recognition

This is a simple application that utilizes python to compute the orientation of a hand, through the help of a web camera, and then feed that rotational information into two [vJoy](https://github.com/shauleiz/vJoy/) joystick axes, namely the X and Y.

This can have several utilities, but the main purpose of this program is to replace HOTAS sets or other peripherals with only hands. 

I am currently able to do a very short duration aerial refueling with the current algorithm. 

Demonstration: https://www.youtube.com/watch?v=u8fZfTvuQq0

# Modules Used
[cv2](https://pypi.org/project/opencv-python/) for operating the camera.
[PyVjoy](https://github.com/tidzo/pyvjoy) for linking up with vJoy.
[mediapipe](https://google.github.io/mediapipe/getting_started/python.html) for tracking the hand, and for cartesian information regarding the hand features.

# Future Ideas

- Custom machine learning algorithm to identify unique user hands.
- A GUI system.
- Support for throttle movement by computing the position of the unused hand.
- Universal calibration system such that the camera can be placed in any orientation and the user can adjust what is recognized as pitch, yaw and roll(currently the camera has to be facing to the left of the hands).
