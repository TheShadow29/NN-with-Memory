import random
import numpy as np
from collections import deque
import keras.backend as K
from keras.models import Model, load_model
from keras.layers import Dense, Conv2D, Flatten, Input, Activation, Lambda, Reshape, Dot, Concatenate
from keras.optimizers import RMSprop
import pickle

from util import gym_util


class MQNAgent:
    def __init__(self, state_size, action_size, gamma, epsilon_init=1.0, epsilon_min=0.1,
                 exploration_steps=1000000, memory_size=40000, init_replay_size=20000,
                 learning_rate=0.00025, momentum=0.95, min_grad=0.01,
                 nn_mem_size=256):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.init_replay_size = init_replay_size
        self.gamma = gamma  # discount rate
        self.epsilon = epsilon_init  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decrease = (epsilon_init - epsilon_min) / exploration_steps
        self.model = self._build_model(nn_mem_size, learning_rate, momentum, min_grad)
        self.target_model = self._build_model(nn_mem_size, learning_rate, momentum, min_grad)
        self.t = 0

    def _build_model(self, nn_mem_size, learning_rate, momentum, min_grad):
        frame_input = Input(shape=self.state_size)
        cnn_features = Conv2D(32, (8, 8), strides=(4, 4), activation='relu')(frame_input)
        cnn_features = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(cnn_features)
        cnn_features = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(cnn_features)
        hist = Lambda(lambda x: K.permute_dimensions(x[:, :, :, 0:-1], (0, 3, 1, 2)))(cnn_features)
        e = Lambda(lambda x: x[:, :, :, -1])(cnn_features)
        hist = Reshape((self.state_size[2] - 1, -1))(hist)
        e = Flatten()(e)
        h = Dense(nn_mem_size, activation='linear')(e)
        m_val = Dense(nn_mem_size, activation='linear')(hist)
        m_key = Dense(nn_mem_size, activation='linear')(hist)
        hm = Dot(axes=-1)([m_key, h])
        p = Activation(activation='softmax')(hm)
        o = Dot(axes=1)([p, m_val])
        ho = Concatenate(axis=1)([h, o])
        q = Dense(512, activation='relu')(ho)
        q = Dense(self.action_size, activation='linear')(q)

        model = Model(inputs=frame_input, outputs=q)
        model.compile(loss=gym_util.huber_loss, optimizer=RMSprop(lr=learning_rate, rho=momentum, epsilon=min_grad))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        self.t += 1
        if self.epsilon > self.epsilon_min and self.t >= self.init_replay_size:
            self.epsilon -= self.epsilon_decrease

        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size), 0

        state = np.float32(np.array(state) / 255.0)
        if len(state.shape) == 3:
            state = state.reshape(tuple([1]) + state.shape)
        act_values = self.model.predict(state)

        return np.argmax(act_values[0]), np.max(act_values[0])  # returns action, Q value

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*minibatch)
        state_batch = np.float32(np.array(state_batch) / 255.0)
        next_state_batch = np.float32(np.array(next_state_batch) / 255.0)
        done_batch = np.array(done_batch) + 0

        q_next_state = self.target_model.predict(next_state_batch)
        y_batch = reward_batch + (1 - done_batch) * self.gamma * np.max(q_next_state, axis=1)
        q_cur = self.model.predict(state_batch)
        for i in range(q_cur.shape[0]):
            q_cur[i, action_batch[i]] = y_batch[i]
        self.model.fit(state_batch, q_cur, epochs=5, verbose=0)

    def update_target(self):
        self.target_model.set_weights(self.model.get_weights())

    def load(self, name):
        with open(name + 'config', 'rb') as f:
            self.state_size = pickle.load(f)
            self.action_size = pickle.load(f)
            self.memory = pickle.load(f)
            self.gamma = pickle.load(f)
            self.epsilon = pickle.load(f)
            self.epsilon_min = pickle.load(f)
            self.epsilon_decrease = pickle.load(f)
            self.t = pickle.load(f)

        self.model = load_model(name + 'model', custom_objects={'huber_loss': gym_util.huber_loss})
        self.target_model = load_model(name + 'target', custom_objects={'huber_loss': gym_util.huber_loss})

    def save(self, name):
        with open(name + 'config', 'wb') as f:
            pickle.dump(self.state_size, f)
            pickle.dump(self.action_size, f)
            pickle.dump(self.memory, f)
            pickle.dump(self.gamma, f)
            pickle.dump(self.epsilon, f)
            pickle.dump(self.epsilon_min, f)
            pickle.dump(self.epsilon_decrease, f)
            pickle.dump(self.t, f)

        self.model.save(name + 'model')
        self.target_model.save(name + 'target')
