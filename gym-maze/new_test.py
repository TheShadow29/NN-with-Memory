import numpy as np

import gym
import gym_maze

if __name__ == '__main__':
    env = gym.make("maze-random-10x10-plus-v0")
    env.reset()
    env.render()
