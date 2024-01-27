# Author: Gyan Tatiya
# Email: Gyan.Tatiya@tufts.edu

import json
import random
import socket

from env import SupermarketEnv
from utils import recv_socket_data


if __name__ == "__main__":
    # Initial Parameters
    counter_y_distance = 0
    distance_cart = 18
    # Make the env
    # env_id = 'Supermarket-v0'
    # env = gym.make(env_id)

    action_commands = ['NOP', 'NORTH', 'SOUTH', 'EAST', 'WEST', 'TOGGLE_CART', 'INTERACT']

    print("action_commands: ", action_commands)

    # Connect to Supermarket
    HOST = '127.0.0.1'
    PORT = 9000
    sock_game = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_game.connect((HOST, PORT))

    while True:
        # action = str(random.randint(0, 1))
        # action += " " + random.choice(action_commands)  # random action

        # assume this is the only agent in the game
        # action = "0 " + random.choice(action_commands)

        # Walking south
        while counter_y_distance <= distance_cart:
            counter_y_distance += 1
            action = "0 " + action_commands[2]
            print("Sending action: ", action)
            sock_game.send(str.encode(action))  # send action to env

            output = recv_socket_data(sock_game)  # get observation from env
            output = json.loads(output)

            print("JSON: ", output)
        counter_y_distance = 0

        # Take cart
        action = "0 " + action_commands[6]
        print("Sending action: ", action)
        sock_game.send(str.encode(action))  # send action to env

        output = recv_socket_data(sock_game)  # get observation from env
        output = json.loads(output)

        print("JSON: ", output)

        # Walking north
        while counter_y_distance <= distance_cart:
            counter_y_distance += 1
            action = "0 " + action_commands[1]
            print("Sending action: ", action)
            sock_game.send(str.encode(action))  # send action to env

            output = recv_socket_data(sock_game)  # get observation from env
            output = json.loads(output)

            print("JSON: ", output)
        counter_y_distance = 0

        # Walking out
        while True:
            action = "0 " + action_commands[4]
            print("Sending action: ", action)
            sock_game.send(str.encode(action))  # send action to env

            output = recv_socket_data(sock_game)  # get observation from env
            output = json.loads(output)

            print("JSON: ", output)