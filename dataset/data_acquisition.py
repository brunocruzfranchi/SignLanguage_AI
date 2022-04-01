"""
This file is intend to be used for data acquisition. Later this
information is going to be separated in train and test set.
"""
import os.path

import cv2
import numpy as np
import mediapipe as mp
from source.functions import mediapipe_detection, draw_styled_landmarks_single_hand, extract_keypoints_single_hand

if __name__ == '__main__':

    # Mediapipe variables
    mp_hands = mp.solutions.hands  # Hands Model
    mp_drawing = mp.solutions.drawing_utils  # Drawing utilities

    DATA_PATH = os.path.join("MP_Data")

    # Actions to be acquired and detected
    actions = np.array(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
                        "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"])

    # Thirty videos worth of data
    no_sequence = 30

    # Video are going to be 30 frames in length
    sequence_length = 30

    # Access to my webcam
    camera_port = 0
    cap = cv2.VideoCapture(camera_port, cv2.CAP_DSHOW)

    with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:

        # Loop through actions
        for action in actions:
            for sequence in range(no_sequence):
                for frame_num in range(sequence_length):

                    # Read Camera Feed
                    ret, frame = cap.read()

                    # Make detection
                    image, results = mediapipe_detection(frame, hands)

                    # Draw Landmarks
                    draw_styled_landmarks_single_hand(image, results)

                    if frame_num == 0:
                        cv2.putText(image, 'Starting Collections', (120, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15, 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                        # Show to screen
                        cv2.imshow('Webcam feed', image)
                        cv2.waitKey(2000)
                    else:
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15, 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.imshow('Webcam feed', image)

                    # Export keypoints
                    keypoints = extract_keypoints_single_hand(results)
                    npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                    np.save(npy_path, keypoints)

                    # Break
                    if cv2.waitKey(10) & 0xFF == ord('c'):
                        break

        cv2.destroyAllWindows()
