"""
From Sentdex RL Series
"""

import gym
import numpy as np
import matplotlib.pyplot as plt


LEARNING_RATE = 0.1
DISCOUNT = 0.95

EPISODES = 50000
SHOW_EVERY = 200

INITIAL_EPSILON = 0.5
EPSILON_DECAY = 0.9
EPSILON_DECAY_FACTOR = 5000
#EPSILON_START_DECAY = 1
#EPSILON_END_DECAY = EPISODES // 2
#epsilon_decay_value = epsilon / (EPSILON_END_DECAY - EPSILON_START_DECAY)

env = gym.make("MountainCar-v0")
env.reset()

print(f"Observation high values: {env.observation_space.high}")
print(f"Observation low values: {env.observation_space.low}")
print(f"Number of actions: {env.action_space.n}")

DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))
print(f"q table shape: {q_table.shape}")

ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': [], 'epsilon': []}


def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(np.int))


for episode in range(EPISODES):
    episode_reward = 0
    if (episode + 1) % SHOW_EVERY == 0 and episode != 0:
        #print(f"Episode: {episode+1}, at epsilon {epsilon}")
        render = False
    else:
        render = False
    discrete_state = get_discrete_state(env.reset())

    epsilon = INITIAL_EPSILON * np.exp(-EPSILON_DECAY * (episode / EPSILON_DECAY_FACTOR))

    done = False
    while not done:
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, 3)
        new_state, reward, done, _ = env.step(action)
        episode_reward += reward
        new_discrete_state = get_discrete_state(new_state)
        if render:
            env.render()
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action, )]
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state + (action, )] = new_q
        elif new_state[0] >= env.goal_position:
            #print(f"We made it on episode {episode+1}")
            q_table[discrete_state + (action, )] = 0

        discrete_state = new_discrete_state

    #if EPSILON_END_DECAY >= episode >= EPSILON_START_DECAY:
        #epsilon -= epsilon_decay_value

    ep_rewards.append(episode_reward)

    if not episode % SHOW_EVERY and episode != 0:
        average_reward = sum(ep_rewards[-SHOW_EVERY:]) / len(ep_rewards[-SHOW_EVERY:])
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['min'].append(min(ep_rewards[-SHOW_EVERY:]))
        aggr_ep_rewards['max'].append(max(ep_rewards[-SHOW_EVERY:]))
        aggr_ep_rewards['epsilon'].append(epsilon)

        print(f"Episode: {episode}, "
              f"avg: {aggr_ep_rewards['avg'][-1]}, "
              f"min: {aggr_ep_rewards['min'][-1]}, "
              f"max: {aggr_ep_rewards['max'][-1]}, "
              f"epsilon: {aggr_ep_rewards['epsilon'][-1]}")


env.close()

fig, ax = plt.subplots(2, 1, figsize=(15, 10))
#plt.figure()
ax[0].plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label='min')
ax[0].plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label='avg')
ax[0].plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label='max')
ax[0].legend(loc=0)

#plt.figure()
ax[1].plot(aggr_ep_rewards['ep'], aggr_ep_rewards['epsilon'], label='epsilon')
ax[1].legend(loc=1)

plt.show()