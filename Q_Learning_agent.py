import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import json  # Import json for dictionary serialization


class QLAgent:
    # here are some default parameters, you can use different ones
    def __init__(self, action_space, alpha=0.5, gamma=0.1, epsilon=0.1, mini_epsilon=0.01, decay=0.999):
        self.action_space = action_space  # numbers of actions
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploit vs. explore probability
        self.mini_epsilon = mini_epsilon  # threshold for stopping the decay
        self.decay = decay  # value to decay the epsilon over time
        self.qtable = self.save_qtable()  # generate the initial table

    def trans(self, state, granularity=0.2):
        # You should design a function to transform the huge state into a learnable state for the agent
        # It should be simple but also contains enough information for the agent to learn
        if state['observation'] is not None:
            x_pos = state['observation']['players'][0]['position'][0]
            y_pos = state['observation']['players'][0]['position'][0]
            curr_cart = state['observation']['players'][0]['curr_cart']
        else:
            x_pos = 1
            y_pos = 13
            curr_cart = -1
        encode_state = self.state_encode(x_pos, y_pos, curr_cart)
        return encode_state

    def learning(self, action, rwd, state, next_state):
        # implement the Q-learning function
        current_state = self.trans(state)
        next_state = self.trans(next_state)

        old_val = self.qtable[current_state][action]
        next_max = max(self.qtable[next_state])

        new_val = (1 - self.alpha) * old_value + self.alpha * (rwd + self.gamma * next_max)

        self.qtable[current_state][action] = new_val

    def choose_action(self, state):
        # implement the action selection for the fully trained agent
        if np.random.uniform() < self.epsilon:
            # Choose a random action
            action = np.random.choice(self.action_space)
        else:
            # Choose the action with the highest Q-value
            state = self.trans(state)
            if state not in self.qtable[state]:
                # If state not encountered before, choose a random action
                action = np.random.choice(self.action_space)
            else:
                action = np.random.choice(self.action_space)
        return action

    def high_q(self, state):
        q_value = []

        for i in range(self.action_space):
            temp_q = self.qtable.at[state, i]
            q_value.append(temp_q)
        print(q_value)
        return np.argmax(q_value)

    def state_encode(self, x_pos, y_pos, cart):
        x = round(x_pos, 1)
        y = round(y_pos, 1)
        encode = round(x * 100000 + y * 100)
        if cart == 1:
            encode = round(encode + 1)
        elif cart == -1:
            pass
        return encode

    def save_qtable(self):

        qtable = {}
        x_min = 1
        y_min = 2
        x_max = 20
        y_max = 24
        granularity = 0.2

        x = x_min

        while (x < x_max):
            y = y_min
            while (y < y_max):
                state = self.state_encode(x, y, 1)
                qtable.update({state: [0, 0, 0, 0, 0]})
                state = self.state_encode(x, y, -1)
                qtable.update({state: [0, 0, 0, 0, 0]})
                y = y + granularity
            x = x + granularity

        file = open("qtable_test.txt", "w")
        for k, v in qtable.items():
            file.writelines(f'{k} {v[0]} {v[1]} {v[2]} {v[3]} {v[4]}\n')
        return qtable
