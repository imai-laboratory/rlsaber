import cv2
import numpy as np


def atari_preprocess(image, shape):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    state = cv2.resize(gray, (210, 160))
    state = cv2.resize(state, (84, 110))
    state = state[18:102, :]
    return state
