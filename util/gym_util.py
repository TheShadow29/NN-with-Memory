import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize
import tensorflow as tf
import keras.backend as K


def init_state(obs, obs_shape, state_len):
    if len(obs_shape) == 2:
        processed_obs = np.uint8(resize(rgb2gray(obs), obs_shape) * 255)
        processed_obs = processed_obs.reshape((obs_shape[0], obs_shape[1]))
        state = [processed_obs for _ in range(state_len)]
        state = np.stack(state, axis=2)
    else:
        state = [obs for _ in range(state_len)]
        state = np.stack(state, axis=0)
    return state


def add_obs(state, obs, obs_shape):
    if len(obs_shape) == 2:
        processed_obs = np.uint8(resize(rgb2gray(obs), obs_shape) * 255)
        processed_obs = processed_obs.reshape((obs_shape[0], obs_shape[1], 1))
        next_state = np.append(state[:, :, 1:], processed_obs, axis=2)
    else:
        obs = obs.reshape((1, obs_shape[0]))
        next_state = np.append(state[1:, :], obs, axis=0)
    return next_state


def huber_loss(y_true, y_pred):
    # Clip the error, the loss is quadratic when the error is in (-1, 1), and linear outside of that region
    error = tf.abs(y_true - y_pred)
    quadratic_part = tf.clip_by_value(error, 0.0, 1.0)
    linear_part = error - quadratic_part
    num_act = tf.to_float(K.shape(y_pred)[1])
    loss = tf.multiply(tf.reduce_mean(0.5 * tf.square(quadratic_part) + linear_part), num_act)
    return loss


def huber_loss_1d(y_true, y_pred):
    y_pred = tf.reduce_max(y_pred, axis=1)
    error = tf.abs(y_true - y_pred)
    quadratic_part = tf.clip_by_value(error, 0.0, 1.0)
    linear_part = error - quadratic_part
    loss = tf.reduce_mean(0.5 * tf.square(quadratic_part) + linear_part)
    return loss

