# This file is tend to learn how to use MediaPipe API to acquired specific keypoint to
# train the model

import cv2
import numpy as np
import mediapipe as mp


def mediapipe_detection(image, model):
    """
    This function is able to receive an image to converted it to RGB spectrum.
    Then the detection is made and finally the image is converted from RGB to BGR
    :param image: frame to be use for detection
    :param model: mediapipe holistic model used
    :return: image and results
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Color Conversion
    image.flags.writeable = False  # Image is no longer writeable. This helps to save memory
    results = model.process(image)  # Make prediction with model
    image.flags.writeable = True  # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Color Conversion
    return image, results


def draw_landmarks(image, results):
    """
    This function is used to draw the landmarks of each detection
    :param image:
    :param results:
    :return:
    """
    mp.solutions.drawing_utils.draw_landmarks(image, results.left_hand_landmarks,
                                              mp.solutions.holistic.HAND_CONNECTIONS)
    mp.solutions.drawing_utils.draw_landmarks(image, results.right_hand_landmarks,
                                              mp.solutions.holistic.HAND_CONNECTIONS)


# Hands Landmarks is an array of (21,3) and for the NN we are going to
# use a flatted array of (63,). So if there is not a hand in frame to be
# detected, we need an array of size (63,) filled with 0.


def draw_styled_landmarks_both_hands(image, results):
    """
    This function is used to draw the landmarks of each detection with a certain style
    for both hands
    :param image:
    :param results: contains the spacial points from the model detection
    :return:
    """
    # Verify if there is any detection made by the model
    if results.multi_hand_landmarks:
        # For each hand detected, the landmarks are drawn
        # In this case we are going to use just one hand
        for hand_landmarks in results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                image,
                hand_landmarks,
                mp.solutions.hands.HAND_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                mp.solutions.drawing_styles.get_default_hand_connections_style())


def draw_styled_landmarks_single_hand(image, results):
    """
    This function is used to draw the landmarks of each detection with a certain style
    for a single hand
    :param image:
    :param results: contains the spacial points from the model detection
    :return:
    """
    # Verify if there is any detection made by the model
    if results.multi_hand_landmarks:
        # For each hand detected, the landmarks are drawn
        # In this case we are going to use just one hand
        mp.solutions.drawing_utils.draw_landmarks(
            image,
            results.multi_hand_landmarks[0],
            mp.solutions.hands.HAND_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
            mp.solutions.drawing_styles.get_default_hand_connections_style())


def extract_keypoints_both_hands(results):
    if results.multi_hand_landmarks:
        # For each hand detected, the landmarks are drawn
        for hand_landmarks in results.multi_hand_landmarks:
            h = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmarks]).flatten() \
                if hand_landmarks else np.zeros((63,))


def extract_keypoints_single_hand(results):
    if results.multi_hand_landmarks:
        hand = np.array([[res.x, res.y, res.z] for res in results.multi_hand_landmarks[0].landmark]).flatten() if \
            results.multi_hand_landmarks else np.zeros((63,))
    else:
        hand = np.zeros((63,))
    return hand


if __name__ == '__main__':

    # Mediapipe variables
    mp_hands = mp.solutions.hands  # Hands Model
    mp_drawing = mp.solutions.drawing_utils  # Drawing utilities

    # Access to my webcam
    camera_port = 0
    cap = cv2.VideoCapture(camera_port,cv2.CAP_DSHOW)
    with mp_hands.Hands(model_complexity=0,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            # Read Camera Feed
            ret, frame = cap.read()

            # Make detection
            image, results = mediapipe_detection(frame, hands)
            draw_styled_landmarks_single_hand(image, results)

            print(extract_keypoints_single_hand(results))
            # Show to screen
            cv2.imshow("Webcam Feed", cv2.flip(image, 1))
            # Break condition
            if cv2.waitKey(10) & 0xFF == ord('c'):
                break

        cv2.destroyAllWindows()
