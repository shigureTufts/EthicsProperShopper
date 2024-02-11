import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import json  # Import json for dictionary serialization

class QLAgent:
    # here are some default parameters, you can use different ones
    def __init__(self, action_space, alpha=0.5, gamma=0.8, epsilon=0.1, mini_epsilon=0.01, decay=0.999):
        self.action_space = action_space  # numbers of actions
        self.alpha = alpha                # learning rate
        self.gamma = gamma                # discount factor
        self.epsilon = epsilon            # exploit vs. explore probability
        self.mini_epsilon = mini_epsilon  # threshold for stopping the decay
        self.decay = decay                # value to decay the epsilon over time
        self.qtable = pd.DataFrame(columns=[i for i in range(self.action_space)])  # generate the initial table
    
    def trans(self, state, granularity=0.5):
        # You should design a function to transform the huge state into a learnable state for the agent
        # It should be simple but also contains enough information for the agent to learn
        state = [float(s) if isinstance(s, (int, float, str)) and s.replace('.', '', 1).isdigit() else 0 for s in state]

        transformed_state = [int(s / granularity) * granularity for s in state]
        return tuple(transformed_state)
        
    def learning(self, action, rwd, state, next_state):
        # implement the Q-learning function
        state = self.trans(state)
        next_state = self.trans(next_state)
        if state not in self.qtable.index:
            self.qtable.loc[state] = [0] * self.action_space
        if next_state not in self.qtable.index:
            self.qtable.loc[next_state] = [0] * self.action_space
        q_predict = self.qtable.loc[state, action]
        q_target = rwd + self.gamma * self.qtable.loc[next_state, :].max()
        self.qtable.loc[state, action] += self.alpha * (q_target - q_predict)

    def choose_action(self, state):
        # implement the action selection for the fully trained agent
        if np.random.uniform() < self.epsilon:
            # Choose a random action
            action = np.random.choice(self.action_space)
        else:
            # Choose the action with the highest Q-value
            state = self.trans(state)
            if state not in self.qtable.index:
                # If state not encountered before, choose a random action
                action = np.random.choice(self.action_space)
            else:
                action = self.qtable.loc[state, :].idxmax()
        return action