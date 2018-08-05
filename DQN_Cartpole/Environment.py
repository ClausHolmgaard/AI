import gym
from gym import wrappers
import os
import pickle


class Environment(object):

    def __init__(self, problem):
        self.problem = problem
        self.env = gym.make(problem)
        #env_tmp_path = os.path.join('/tmp/', problem)
        #self.env = wrappers.Monitor(self.env, env_tmp_path, video_callable=False, force=True)

        self.reward_history = []

    def run(self, agent):
        s = self.env.reset()
        R = 0

        while True:            
            #self.env.render()

            a = agent.act(s)

            s_, r, done, info = self.env.step(a)

            if done: # terminal state
                s_ = None

            agent.observe((s, a, r, s_))
            agent.replay()            

            s = s_
            R += r

            if done:
                break

        self.reward_history.append(R)
        print(f"Total reward: {R}", end='')
        if len(self.reward_history) % 100 == 0:
            print(f"  - Saving at {len(self.reward_history)} episodes")
            self.save_history('Results/cartpole-reward-history.pickle')
            agent.brain.model.save('Results/cartpole-basic.h5')
        else:
            print()

    def save_history(self, file_name):
        savefile = open(file_name, 'wb')
        pickle.dump(self.reward_history, savefile)
        savefile.close()