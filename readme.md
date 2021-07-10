## Pose estimation using MoveNet by Tensorflow

This repository contains python code for experimenting with the MoveNet Pose estimation Model on the the TensorFlow hub. There are two models: The **MoveNet Lightening** and **MoveNet Thunder**. 

The script assumes you have a webcam attached. You may have to change the video capture source for capturing the images.

The MoveNet model detect the following Landmarks:
**Landmarks nose left eye, right eye, left ear, right ear, left shoulder, right shoulder, left elbow, right elbow, left wrist, right wrist, left hip, right hip, left knee, right knee, left ankle, right ankle**

SOURCE: https://tfhub.dev/s?q=movenet