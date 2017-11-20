import random
import numpy as np
from collections import deque
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM
from keras.optimizers import RMSprop
import pickle
import pdb

from util import gym_util


class SimpleDRQNAgent:
    def __init__(self, state_size, action_size, gamma, epsilon_init=1.0, epsilon_min=0.1,
                 exploration_steps=1000000, memory_size=40000, init_replay_size=20000,
                 learning_rate=0.00025, momentum=0.95, min_grad=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.init_replay_size = init_replay_size
        self.gamma = gamma  # discount rate
        self.epsilon = epsilon_init  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decrease = (epsilon_init - epsilon_min) / exploration_steps
        self.model = self._build_model(learning_rate, momentum, min_grad)
        self.target_model = self._build_model(learning_rate, momentum, min_grad)
        self.t = 0

    def _build_model(self, learning_rate, momentum, min_grad):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(32, activation='relu', input_shape=self.state_size))
        model.add(LSTM(32, activation='tanh'))
        model.add(Dense(self.action_size, activation='linear'))
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

        if len(state.shape) == 2:
            state = state.reshape((1, state.shape[0], state.shape[1]))
        act_values = self.model.predict(state)

        return np.argmax(act_values[0]), np.max(act_values[0])  # returns action, Q value

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*minibatch)
        state_batch = np.array(state_batch)
        action_batch = np.array(action_batch)
        next_state_batch = np.array(next_state_batch)
        reward_batch = np.array(reward_batch)
        done_batch = np.array(done_batch) + 0

        q_next_state = self.target_model.predict(next_state_batch)
        y_batch = reward_batch + (1 - done_batch) * self.gamma * np.max(q_next_state, axis=1)

        q_cur = self.model.predict(state_batch)
        for i in range(q_cur.shape[0]):
            q_cur[i, action_batch[i]] = y_batch[i]
        self.model.fit(state_batch, q_cur, epochs=1, verbose=0)

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
