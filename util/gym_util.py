import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize
import tensorflow as tf


def init_state(obs, obs_shape, state_len):
    processed_obs = np.uint8(resize(rgb2gray(obs), obs_shape) * 255)
    processed_obs = processed_obs.reshape((obs_shape[0], obs_shape[1]))
    state = [processed_obs for _ in range(state_len)]
    state = np.stack(state, axis=2)
    return state


def add_obs(state, obs, obs_shape):
    processed_obs = np.uint8(resize(rgb2gray(obs), obs_shape) * 255)
    processed_obs = processed_obs.reshape((obs_shape[0], obs_shape[1], 1))
    next_state = np.append(state[:, :, 1:], processed_obs, axis=2)
    return next_state


def huber_loss(y_true, y_pred):
    # Clip the error, the loss is quadratic when the error is in (-1, 1), and linear outside of that region
    error = tf.abs(y_true - y_pred)
    quadratic_part = tf.clip_by_value(error, 0.0, 1.0)
    linear_part = error - quadratic_part
    loss = tf.reduce_mean(0.5 * tf.square(quadratic_part) + linear_part)
    return loss
