import numpy as np

import gym
import gym_maze

if __name__ == '__main__':
    # env = gym.make("maze-random-10x10-plus-v0")
    env = gym.make('maze-test-v0')
    print(env.reset())
    env.render()
    print(env.step(0))
    # while True:
    # pass
