from models.dqn import SimpleDQNAgent
from models.drqn import SimpleDRQNAgent
from models.mqn import SimpleMQNAgent
from models.rmqn import SimpleRMQNAgent
from util import gym_util

import gym
import gym_maze
from tensorboard_logger import configure, log_value
import time
import os
import numpy as np
import pdb

if __name__ == "__main__":
    env_name = 'maze-test-v0'
    env = gym.make(env_name)
    obs_size = env.observation_space.shape[0]  # Size of observation from environment
    # max_obs = int(env.observation_space.high[0])
    state_length = 50  # Number of most recent frames to produce the input to the network
    action_size = env.action_space.n  # Number of actions
    gamma = 0.99  # Discount factor
    max_episode_length = 10000  # Time after which an episode is terminated
    n_episodes = 15000  # Number of episodes the agent plays
    agent_model = 'drqn'

    epsilon_init = 1.0  # Initial value of epsilon in epsilon-greedy
    epsilon_min = 0.1  # Minimum value of epsilon in epsilon-greedy
    exploration_steps = 1000000  # Number of frames over which the initial value of epsilon is linearly annealed

    init_replay_size = 50000  # Number of steps to populate the replay memory before training starts
    replay_size = 100000  # Number of replay memory the agent uses for training
    batch_size = 32  # Mini batch size
    target_update_interval = 10000  # The frequency with which the target network is updated
    train_interval = 4  # The agent selects 4 actions between successive updates

    learning_rate = 0.00025  # Learning rate used by RMSProp
    momentum = 0.95  # Momentum used by RMSProp
    min_grad = 0.01  # Constant added to the squared gradient in the denominator of the RMSProp update

    save_interval = 30000  # The frequency with which the network is saved
    load_network = False
    load_network_path = 'saved_networks/' + env_name + '/' + '1510501206'
    save_network_path = 'saved_networks/' + env_name + '/' + str(time.time()).split('.')[0]
    save_summary_path = 'summary/' + env_name + '/' + str(time.time()).split('.')[0]
    os.makedirs(os.path.dirname(save_network_path), exist_ok=True)
    configure(save_summary_path)

    if agent_model == 'dqn':
        agent = SimpleDQNAgent((state_length, obs_size), action_size, gamma,
                               epsilon_init=epsilon_init, epsilon_min=epsilon_min, exploration_steps=exploration_steps,
                               memory_size=replay_size, init_replay_size=init_replay_size, learning_rate=learning_rate,
                               momentum=momentum, min_grad=min_grad)
    elif agent_model == 'drqn':
        agent = SimpleDRQNAgent((state_length, obs_size), action_size, gamma,
                                epsilon_init=epsilon_init, epsilon_min=epsilon_min, exploration_steps=exploration_steps,
                                memory_size=replay_size, init_replay_size=init_replay_size, learning_rate=learning_rate,
                                momentum=momentum, min_grad=min_grad)
    elif agent_model == 'mqn':
        agent = SimpleMQNAgent((state_length, obs_size), action_size, gamma,
                               epsilon_init=epsilon_init, epsilon_min=epsilon_min, exploration_steps=exploration_steps,
                               memory_size=replay_size, init_replay_size=init_replay_size, learning_rate=learning_rate,
                               momentum=momentum, min_grad=min_grad)
    elif agent_model == 'rmqn':
        agent = SimpleRMQNAgent((state_length, obs_size), action_size, gamma,
                                epsilon_init=epsilon_init, epsilon_min=epsilon_min, exploration_steps=exploration_steps,
                                memory_size=replay_size, init_replay_size=init_replay_size, learning_rate=learning_rate,
                                momentum=momentum, min_grad=min_grad)

    if load_network:
        agent.load(load_network_path)

    done = False
    # timeout_spree = 0

    for e in range(n_episodes):
        obs = env.reset()
        total_reward = 0
        total_max_q = 0
        episode_duration = 0
        state = gym_util.init_state(obs, obs_size, state_length)
        # timeout_spree += 1
        for time in range(max_episode_length):
            # env.render()
            episode_duration += 1
            action, q_value = agent.act(state)
            action = int(action)
            total_max_q += q_value
            # pdb.set_trace()
            next_obs, reward, done, _ = env.step(action)
            # reward = np.clip(reward, -1, 1)
            next_state = gym_util.add_obs(state, next_obs, obs_size)
            total_reward += reward
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if agent.t > init_replay_size:
                # Train network
                if agent.t % train_interval == 0:
                    agent.replay(batch_size)

                # Update target network
                if agent.t % target_update_interval == 0:
                    agent.update_target()

                # Save network
                if agent.t % save_interval == 0:
                    agent.save(save_network_path)

            if done:
                print("episode: {}/{}, score: {}, avg max q: {}, episode duration: {}, e: {:.2}"
                      .format(e, n_episodes, total_reward, total_max_q / episode_duration, episode_duration,
                              agent.epsilon))
                log_value('Episode', e, e)
                log_value('Score', total_reward, e)
                log_value('Avg max Q', total_max_q / episode_duration, e)
                log_value('Episode duration', episode_duration, e)
                log_value('Epsilon', agent.epsilon, e)
                # if episode_duration < 5000:
                #     timeout_spree = 0
                break

        # if timeout_spree >= 20:
        #     print("Ending since agent didn't finish last 20 episodes")
        #     break
