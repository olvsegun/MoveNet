import numpy as np 
import tensorflow as tf 
import cv2 as cv


# Load Model
interpreter = tf.lite.Interpreter(model_path="Models/lite-model_movenet_singlepose_thunder_3.tflite")
interpreter.allocate_tensors()


# Draw Keypoints

def draw_keypoints(frame, keypoints, conf_thresh):
    """
    This function draws circles on the keypoints/landmarks
    on the image.
    Parameters:
        Frame: The image to draw keypoints on
        keypoints: The coordinates of the landmarks normalised
        conf_thresh: the confidence probablility required to draw
    Output:
        cirsles drawn over landmarks on image
    """
    h, w, _ = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [h, w, 1]))
    for landmark in shaped:
        ky, kx, conf = landmark
        if conf > conf_thresh:
            cv.circle(frame, (int(kx), int(ky)), 4, (0, 0, 255), -1)

    
# Draw edges

edges = {
     (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}
def draw_connections(frame, keypoints, edges, conf_thresh=0.4):
    """
    This function draws between keypoints/landmarks
    on the image.
    Parameters:
        Frame: The image to draw keypoints on
        keypoints: The coordinates of the landmarks normalised
        edges: (dict) Possible connection pairs between images
        conf_thresh: the confidence probablility required to draw
    Output:
        line drawn connecting images.
    """
    h, w, _ = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [h, w, 1]))
    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, conf1 = shaped[p1]
        y2, x2, conf2 = shaped[p2]
        if (conf1 >= conf_thresh) & (conf2 >= conf_thresh):
            cv.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255,0,0), 2 )

cap = cv.VideoCapture(1)
while cap.isOpened():
    _, frame = cap.read()
    # Reshape image
    img = frame.copy()
    image = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 256, 256)

    # cast image as tf.float32
    input_image = tf.cast(image, dtype=tf.float32)

    #setup input and output
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # make predictions
    interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
    interpreter.invoke()
    # output
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])

    #Redering
    draw_keypoints(frame, keypoints_with_scores, conf_thresh=0.2)
    draw_connections(frame, keypoints_with_scores, edges, conf_thresh=0.5)

    cv.imshow("Movenet - Thunder", frame)
    if cv.waitKey(10) & 0xFF==ord("q"):
        break
cap.release()
cv.destroyAllWindows()

