# run from ../gym directory

import os
import sys

if not(os.path.abspath('..') in sys.path):
    sys.path.append(os.path.abspath('..'))

#from keras.layers import Flatten, Dense
from tensorflow.python.keras.layers import Dense, Input
#from keras.models import Sequential
from tensorflow.python.keras.models import Sequential
#from keras.optimizers import Adam
from tensorflow.python.keras.optimizers import Adam
from Helpers.TensorboardHelpers import ModifiedTensorBoard
#from tensorflow.python.keras.callbacks import TensorBoard
from collections import deque
#import matplotlib.pyplot as plt
#from tqdm import tqdm
import time
import random
import numpy as np
#import gym

DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 100  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 16  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 1  # Terminal states (end of episodes)

# Environment settings
EPISODES = 2000

# Exploration settings
epsilon = 0.2  # not a constant, going to be decayed
EPSILON_DECAY = 0.999
MIN_EPSILON = 0.001

#  Stats settings
AGGREGATE_STATS_EVERY = 50  # episodes
SHOW_PREVIEW = False

#env = gym.make("MountainCar-v0")
#env.reset()

#print(f"Observation high values: {env.observation_space.high}")
#print(f"Observation low values: {env.observation_space.low}")
#print(f"Number of actions: {env.action_space.n}")

MODEL_NAME = "Dense8"

class DQNAgent:
    def __init__(self, environment, log_dir):

        self.env = environment

        # main model  # gets trained every step
        self.model = self.create_model()

        # Target model this is what we .predict against every step
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.tensorboard = ModifiedTensorBoard(log_dir=f"{log_dir}/{MODEL_NAME}-{int(time.time())}")
        #self.tensorboard = TensorBoard(log_dir=f"{log_dir}/{MODEL_NAME}-{int(time.time())}")

        self.target_update_counter = 0

    def create_model(self):
        model = Sequential()

        #model.add(Conv2D(16, (3, 3), input_shape=env.OBSERVATION_SPACE_VALUES))  # OBSERVATION_SPACE_VALUES = (10, 10, 3) a 10x10 RGB image.
        #model.add(Activation('relu'))
        #model.add(MaxPooling2D(pool_size=(2, 2)))
        #model.add(Dropout(0.2))

        #model.add(Conv2D(256, (3, 3)))
        #model.add(Activation('relu'))
        #model.add(MaxPooling2D(pool_size=(2, 2)))
        #model.add(Dropout(0.2))

        model.add(Input(shape=(2,)))
        model.add(Dense(8))

        #model.add(Dense(64))

        model.add(Dense(self.env.action_space.n, activation='linear'))  # ACTION_SPACE_SIZE = how many choices (9)
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # Trains main network every step during episode
    def train(self, terminal_state, step):

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(X),
                       np.array(y),
                       batch_size=MINIBATCH_SIZE,
                       verbose=0,
                       shuffle=False,
                       callbacks=[self.tensorboard] if terminal_state else None
                       )

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape))[0]


def preprocess(inp):
    # Oh my! Very ugly, should fix...1
    high = np.array([0.6, 0.07])
    low = np.array([-1.2, -0.07])

    res = (inp - low) / (high - low)

    return res

"""
if __name__ == "__main__":
    agent = DQNAgent()

    render = True

    ep_rewards = []
    epsilons = []

    for episode in tqdm(range(EPISODES), ncols=150):
    #for episode in range(EPISODES):
        episode_reward = 0

        # Update tensorboard step every episode
        agent.tensorboard.step = episode

        # Restarting episode - reset episode reward and step number
        episode_reward = 0
        step = 1

        # Reset environment and get initial state
        current_state = preprocess(env.reset())

        # Reset flag and start iterating until episode ends
        done = False
        while not done:

            # This part stays mostly the same, the change is to query a model for Q values
            if np.random.random() > epsilon:
                # Get action from Q table
                action = np.argmax(agent.get_qs(current_state))
            else:
                # Get random action
                action = np.random.randint(0, env.action_space.n)

            new_state, reward, done, _ = env.step(action)
            new_state = preprocess(new_state)

            # Transform new continous state to new discrete state and count reward
            episode_reward += reward

            if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
                env.render()

            # Every step we update replay memory and train main network
            agent.update_replay_memory((current_state, action, reward, new_state, done))
            agent.train(done, step)

            current_state = new_state
            step += 1

        ep_rewards.append(episode_reward)

        # Decay epsilon
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)

        epsilons.append(epsilon)

    plt.figure()
    plt.plot(ep_rewards, label='Rewards')
    plt.legend(loc=0)
    plt.figure()
    plt.plot(epsilons, label='Epsilon')
    plt.legend(loc=1)
    plt.show()

"""