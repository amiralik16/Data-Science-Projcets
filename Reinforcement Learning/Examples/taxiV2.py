import numpy as np
import gym
import random
import os
import time

class TaxiAI:
    def __init__(self):
        self.env = gym.make('Taxi-v2')
        self.qtable = np.zeros((self.env.observation_space.n,self.env.action_space.n))
        #hyperparams
        self.total_episodes = 50000        # Total episodes
        self.learning_rate = 0.7           # Learning rate
        self.max_steps = 100               # Max steps per episode
        self.gamma = 0.618                  # Discounting rate

        # Exploration parameters
        self.epsilon = 1.0                 # Exploration rate
        self.max_epsilon = 1.0             # Exploration probability at start
        self.min_epsilon = 0.01            # Minimum exploration probability 
        self.decay_rate = 0.01             # Exponential decay rate for exploration prob

    def do_explore(self,epsilon):
        r = random.uniform(0,1)
        if r<epsilon:
            return True
        return False

    def train(self):
        for episode in range(self.total_episodes):
            state = self.env.reset()
            done = False
            step = 0
            t_rewards = 0
            for step in range(self.max_steps):
                #choose action <-- do_explore, if True: explore, if False: exploit
                if self.do_explore(self.epsilon):
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.qtable[state])
                #take action
                nstate, reward, done, info = self.env.step(action)
                #update table
                self.qtable[state][action] += self.learning_rate*(reward + self.gamma * max(self.qtable[nstate]) - self.qtable[state][action]) 
                state = nstate
                if done: 
                    break
                
            self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay_rate * episode)

    def ai_test(self):
        state = self.env.reset()
        test_num = 10
        rewards = []
        for _ in range(test_num):
            state = self.env.reset()
            t_rewards = 0
            for step in range(self.max_steps):
                action = np.argmax(self.qtable[state])
                nstate, reward, done, info = self.env.step(action)
                t_rewards += reward
                state = nstate
                if done:
                    self.env.render()
                    rewards.append(t_rewards)
                    
        print(sum(rewards)/test_num)

    def ai_play(self):
        clear = lambda: os.system('cls')
        state = self.env.reset()
        for step in range(self.max_steps):
            clear()
            action = np.argmax(self.qtable[state])
            nstate, reward, done, info = self.env.step(action)
            # t_rewards += reward
            state = nstate
            self.env.render()
            if done:
                break
            time.sleep(1)