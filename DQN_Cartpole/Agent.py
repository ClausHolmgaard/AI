import numpy as np
import random
import math

from Brain import Brain
from Memory import Memory


MEMORY_CAPACITY = 1000000

GAMMA = 0.99

MAX_EPSILON = 1
MIN_EPSILON = 0.01
LAMBDA = 0.001      # speed of decay

BATCH_SIZE = 64
UPDATE_TARGET_FREQUENCY = 1000

class Agent(object):
    def __init__(self, state_count, action_count):
        self.state_count = state_count
        self.action_count = action_count

        self.brain = Brain(state_count, action_count)
        self.memory = Memory(MEMORY_CAPACITY)

        self.epsilon = MAX_EPSILON
        self.steps = 0

    def act(self, s):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_count - 1)
        else:
            return np.argmax(self.brain.predict_one(s))

    def observe(self, samples):
        self.memory.add(samples)

        if self.steps % UPDATE_TARGET_FREQUENCY == 0:
            self.brain.update_target()

        # slowly decrease Epsilon based on our eperience
        self.steps += 1
        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)
    
    def replay(self):   
        batch = self.memory.sample(BATCH_SIZE)
        batchLen = len(batch)

        no_state = np.zeros(self.state_count)

        states = np.array([ o[0] for o in batch ])
        states_ = np.array([ (no_state if o[3] is None else o[3]) for o in batch ])

        p = self.brain.predict(states)
        #p_ = self.brain.predict(states_, target=True)

        p_ = self.brain.predict(states_, target=False)
        pTarget_ = self.brain.predict(states_, target=True)

        x = np.zeros((batchLen, self.state_count))
        y = np.zeros((batchLen, self.action_count))
        
        for i in range(batchLen):
            o = batch[i]
            s = o[0]; a = o[1]; r = o[2]; s_ = o[3]
            
            t = p[i]
            if s_ is None:
                t[a] = r
            else:
                #t[a] = r + GAMMA * np.amax(p_[i])
                t[a] = r + GAMMA * pTarget_[i][np.argmax(p_[i])]  # double DQN

            x[i] = s
            y[i] = t

        self.brain.train(x, y)