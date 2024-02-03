import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import json  # Import json for dictionary serialization

class QLAgent:
    # here are some default parameters, you can use different ones
    def __init__(self, action_space, alpha=0.5, gamma=0.8, epsilon=0.1, mini_epsilon=0.01, decay=0.999):
        self.action_space = action_space 
        self.alpha = alpha               # learning rate
        self.gamma = gamma               # discount factor  
        self.epsilon = epsilon           # exploit vs. explore probability
        self.mini_epsilon = mini_epsilon # threshold for stopping the decay
        self.decay = decay               # value to decay the epsilon over time
        self.qtable = pd.DataFrame(columns=[i for i in range(self.action_space)])  # generate the initial table
    
    def trans(self, state, granularity=0.5):
        # You should design a function to transform the huge state into a learnable state for the agent
        # It should be simple but also contains enough information for the agent to learn
        
        pass
        
    def learning(self, action, rwd, state, next_state):
        # implement the Q-learning function
        state = self.trans(state)
        next_state = self.trans(next_state)

        if next_state not in self.qtable.index:
            self.qtable = self.qtable.append(
                pd.Series([0] * self.action_space, index=self.qtable.columns, name=next_state)
            )

        q_predict = self.qtable.loc[state, action]

        if next_state not in self.qtable.index:
            q_target = rwd
        else:
            q_target = rwd + self.gamma * self.qtable.loc[next_state, :].max()

        self.qtable.loc[state, action] += self.alpha * (q_target - q_predict)

    def choose_action(self, state):
        # implement the action selection for the fully trained agent
        state = self.trans(state)

        # Exploration-exploitation trade-off
        if np.random.uniform(0, 1) < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            action = self.qtable.loc[state, :].idxmax()

        return action
    def decay_epsilon(self):
        if self.epsilon > self.mini_epsilon:
            self.epsilon *= self.decay