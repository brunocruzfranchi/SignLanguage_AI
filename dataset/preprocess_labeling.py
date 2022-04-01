"""

"""

import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


def preprocess_data():
    # Path to get data
    DATA_PATH = os.path.join("MP_Data")

    actions = np.array(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
                        "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"])

    # Thirty videos worth of data
    no_sequence = 30

    # Video are going to be 30 frames in length
    sequence_length = 30

    label_map = {label: num for num, label in enumerate(actions)}

    sequences, labels = [], []

    for action in actions:
        for sequence in range(no_sequence):
            window = []
            for frame_num in range(sequence_length):
                res = np.load(os.path(DATA_PATH, action, str(sequence), "{}.py".format(frame_num)))
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])

    X = np.array(sequences)
    y = to_categorical(labels).astype(int)

    return X, y


def train_test_dataset(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
    return X_train, X_test, y_train, y_test



