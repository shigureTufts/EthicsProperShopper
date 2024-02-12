#Author Hang Yu

import json
import random
import socket

import gymnasium as gym
from env import SupermarketEnv
from utils import recv_socket_data

from Q_Learning_agent import QLAgent  # Make sure to import your QLAgent class
import pickle
import pandas as pd


cart = False
exit_pos = [-0.8, 15.6] # The position of the exit in the environment from [-0.8, 15.6] in x, and y = 15.6
cart_pos_left = [1, 18.5] # The position of the cart in the environment from [1, 2] in x, and y = 18.5
cart_pos_right = [2, 18.5] 

def distance_to_cart(state):
    agent_position = state['observation']['players'][0]['position']
    if agent_position[0] > 1.5:
        cart_distances = [euclidean_distance(agent_position, cart_pos_right)]
    else:
        cart_distances = [euclidean_distance(agent_position, cart_pos_left)]
    return min(cart_distances)

def euclidean_distance(pos1, pos2):
    # Calculate Euclidean distance between two points
    return ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5


def calculate_reward(previous_state, current_state):
    # design your own reward function here
    # You should design a function to calculate the reward for the agent to guide the agent to do the desired task
    cart_state = current_state['observation']['players'][0]['curr_cart']
    agent_position = current_state['observation']['players'][0]['position']
    # move to cart
    if cart_state == -1:
        # Calculate the distance between the agent and the cart
        # handel None type
        distance_to_cart_current = distance_to_cart(current_state)
        if previous_state['observation'] is not None:
            distance_to_cart_previous = distance_to_cart(previous_state)
        else:
            distance_to_cart_previous = 10

        if distance_to_cart_current < distance_to_cart_previous:
            rwd = 10   # reward for reaching the cart
        elif distance_to_cart_current > distance_to_cart_previous:
            rwd = -10  # negative reward for moving away from the cart
        else:
            rwd = -1   # small negative reward for no progress

    # get the cart to exit
    elif cart_state == 1:
        # Design reward based on the distance to the exit
        agent_previous_position = current_state['observation']['players'][0]['position']
        distance_to_exit_current = euclidean_distance(agent_position, exit_pos)
        distance_to_exit_previous = euclidean_distance(agent_previous_position, exit_pos)
        if distance_to_exit_current < distance_to_exit_previous:
            rwd = 10   # reward for reaching the exit
        elif distance_to_exit_current > distance_to_exit_previous:
            rwd = -10  # negative reward for moving away from exit
        elif cart_state == -1:
            rwd = -20  # negative reward for let go of cart
        else:
            rwd = -1   # small negative reward for no progress
    else:
        rwd = 0
    return rwd

if __name__ == "__main__":

    action_commands = ['NOP', 'NORTH', 'SOUTH', 'EAST', 'WEST', 'TOGGLE_CART', 'INTERACT', 'RESET']
    # Initialize Q-learning agent
    action_space = len(action_commands) - 1   # Assuming your action space size is equal to the number of action commands
    agent = QLAgent(action_space)

####################
    #Once you have your agent trained, or you want to continue training from a previous training session, you can load the qtable from a json file
    #agent.qtable = pd.read_json('qtable.json')
####################
    
    
    # Connect to Supermarket
    HOST = '127.0.0.1'
    PORT = 9000
    sock_game = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_game.connect((HOST, PORT))

    training_time = 100
    episode_length = 1000
    for i in range(training_time):
        sock_game.send(str.encode("0 RESET"))  # reset the game
        state = recv_socket_data(sock_game)
        state = json.loads(state)
        cnt = 0
        while not state['gameOver']:
            cnt += 1
            # Choose an action based on the current state
            action_index = agent.choose_action(state)
            action = "0 " + action_commands[action_index]

            print("Sending action: ", action)
            sock_game.send(str.encode(action))  # send action to env

            next_state = recv_socket_data(sock_game)  # get observation from env
            next_state = json.loads(next_state)

            # Define the reward based on the state and next_state
            reward = calculate_reward(state, next_state)  # You need to define this function
            print("------------------------------------------")
            print(reward, action_commands[action_index])
            print("------------------------------------------")
            # Update Q-table
            agent.learning(action_index, reward, state, next_state)

            # Update state
            state = next_state
            agent.qtable.to_json('qtable.json')

            if cnt > episode_length:
                break
        # Additional code for end of episode if needed

    # Close socket connection
    sock_game.close()

