"""
This .py is intended to be used to create the folders for data
acquisition.
"""

import numpy as np
import os

# Path for exported data
DATA_PATH = os.path.join("MP_Data")

# Actions to be acquired and detected
actions = np.array(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
                    "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"])

# Thirty videos worth of data
no_sequence = 30

# Video are going to be 30 frames in length
sequence_length = 30

for action in actions:
    for sequence in range(no_sequence):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass
