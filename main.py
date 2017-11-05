from models.dqn import DQNAgent
from util import gym_util

import gym

if __name__ == "__main__":
    env_name = 'Pong-v0'
    env = gym.make(env_name)
    obs_size = env.observation_space.shape  # Size of observation from environment
    frame_width = 84  # Resized frame width
    frame_height = 84  # Resized frame height
    state_length = 4  # Number of most recent frames to produce the input to the network
    action_size = env.action_space.n  # Number of actions
    gamma = 0.99  # Discount factor
    max_episode_length = 1000000  # Time after which an episode is terminated
    n_episodes = 12000  # Number of episodes the agent plays
    no_op_steps = 30  # Number of initial steps to not take actions

    epsilon_init = 1.0  # Initial value of epsilon in epsilon-greedy
    epsilon_min = 0.1  # Minimum value of epsilon in epsilon-greedy
    exploration_steps = 1000000  # Number of frames over which the initial value of epsilon is linearly annealed

    init_replay_size = 20000  # Number of steps to populate the replay memory before training starts
    replay_size = 40000  # Number of replay memory the agent uses for training
    batch_size = 32  # Mini batch size
    target_update_interval = 10000  # The frequency with which the target network is updated
    train_interval = 4  # The agent selects 4 actions between successive updates

    learning_rate = 0.00025  # Learning rate used by RMSProp
    momentum = 0.95  # Momentum used by RMSProp
    min_grad = 0.01  # Constant added to the squared gradient in the denominator of the RMSProp update

    save_interval = 300000  # The frequency with which the network is saved
    load_network = False
    save_network_path = 'saved_networks/' + env_name
    save_summary_path = 'summary/' + env_name

    agent = DQNAgent((frame_height, frame_width, state_length), action_size, gamma,
                     epsilon_init=epsilon_init, epsilon_min=epsilon_min, exploration_steps=exploration_steps,
                     memory_size=replay_size, init_replay_size=init_replay_size, learning_rate=learning_rate,
                     momentum=momentum, min_grad=min_grad, logdir=save_summary_path)
    if load_network:
        agent.load(save_network_path)
    done = False

    t = 0
    for e in range(n_episodes):
        obs = env.reset()
        total_reward = 0
        total_max_q = 0
        episode_duration = 0
        for time in range(no_op_steps):
            # env.render()
            obs, _, _, _ = env.step(0)
        state = gym_util.init_state(obs, (frame_width, frame_height), state_length)
        for time in range(max_episode_length):
            # env.render()
            t += 1
            episode_duration += 1
            action, q_value = agent.act(state)
            total_max_q += q_value
            next_obs, reward, done, _ = env.step(action)
            next_state = gym_util.add_obs(state, next_obs, (frame_width, frame_height))
            total_reward += reward
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if t > init_replay_size:
                # Train network
                if t % train_interval == 0:
                    agent.replay(batch_size)

                # Update target network
                if t % target_update_interval == 0:
                    agent.update_target()

                # Save network
                if t % save_interval == 0:
                    agent.save(save_network_path)

            if done:
                print("episode: {}/{}, score: {}, avg max q: {}, episode duration: {}, e: {:.2}"
                      .format(e, n_episodes, total_reward, total_max_q / episode_duration, episode_duration,
                              agent.epsilon))
                break
