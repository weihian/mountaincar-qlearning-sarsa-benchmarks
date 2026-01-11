import math
import random
import time

# from time import sleep
import gym
import numpy as np
from mpmath import *

# Initialize the "Cart-Pole" environment
# env = gym.make('MountainCar-v0', render_mode = "human")
env = gym.make('MountainCar-v0')

# Initializing the random number generator
np.random.seed(int(time.time()))

# Defining the environment related constants

# Number of discrete states (bucket) per state dimension
NUM_BUCKETS = (19, 29)  # (x, x', theta, theta')
# Number of discrete actions
NUM_ACTIONS = 3  # (left, right)
# Bounds for each discrete state
STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))
STATE_BOUNDS[0] = (-1.2, 0.6, 19)
STATE_BOUNDS[1] = (-0.07, 0.07, 29)
# Index of the action
ACTION_INDEX = len(NUM_BUCKETS)

# Creating a Q-Table for each state-action pair
q_table = np.zeros(NUM_BUCKETS + (NUM_ACTIONS,))

# Learning related constants
# Continue here
MIN_EXPLORE_RATE = 0.2
MIN_LEARNING_RATE = 0.4  # 0.1-0.3

# Defining the simulation related constants
NUM_TRAIN_EPISODES = 1000  # 1000
MAX_TRAIN_T = 200
STREAK_TO_END = 120
SOLVED_T = 199
VERBOSE = False


def train():
    # Instantiating the learning related parameters
    learning_rate = get_learning_rate(0)
    explore_rate = get_explore_rate(0)
    discount_factor = 0.08  # since the world is unchanging
    win = 0

    num_train_streaks = 0

    for episode in range(NUM_TRAIN_EPISODES):

        # Reset the environment
        obv_0, _ = env.reset()

        # the initial state
        state_0 = state_to_bucket(obv_0)

        for t in range(MAX_TRAIN_T):
            env.render()

            # Select an action
            action_a = select_action(state_0, explore_rate)

            action_b = select_action(state_0, explore_rate)

            selection = random.randint(0, 1)

            if selection < mp.exp((q_table[state_0 + (action_a,)] - q_table[state_0 + (action_b,)])/(t+1)):
                action_0 = action_a
            else:
                action_0 = action_b

            # Execute the action
            obv_1, reward, done, _, _ = env.step(action_0)

            # Observe the result
            state_1 = state_to_bucket(obv_1)

            # kinetic energy and potential energy reward function
            reward = reward + ((20 * abs(obv_1[1])) + (20 * (abs(obv_1[0] - obv_0[0])))) / 2

            # Update the Q based on the result
            best_q = np.amax(q_table[state_1])
            q_table[state_0 + (action_0,)] += learning_rate * (
                    reward + discount_factor * best_q - q_table[state_0 + (action_0,)])

            # Setting up for the next iteration
            state_0 = state_1
            obv_0 = obv_1

            # Print data
            if VERBOSE:
                print("\nEpisode = %d" % episode)
                print("t = %d" % t)
                print("Reward: %f" % reward)
                print("Explore rate: %f" % explore_rate)
                print("Learning rate: %f" % learning_rate)
                print("Streaks: %d" % num_train_streaks)

                print("")

            if done:
                win += 1
                print("Episode %d finished after %f time steps" % (episode, t))
                print("streaks")
                print(num_train_streaks)
                if t >= SOLVED_T:
                    num_train_streaks += 1
                else:
                    num_train_streaks = 0
                break

            # sleep(0.25)

        # print("streaks")
        # print(num_train_streaks)

        # It's considered done when it's solved over 120 times consecutively
        if num_train_streaks > STREAK_TO_END:
            break

        # Update parameters
        explore_rate = get_explore_rate(episode)
        learning_rate = get_learning_rate(episode)
    print("times of win")
    print(win)


def select_action(state, explore_rate):
    # Select a random action
    if random.random() < explore_rate:
        action = random.randint(0, 2)
    # Select the action with the highest q
    else:
        action = np.argmax(q_table[state])
    return action


def get_explore_rate(t):
    return max(MIN_EXPLORE_RATE, min(0.5, 1.0 - math.log10((t + 1) / 25)))


def get_learning_rate(t):
    return max(MIN_LEARNING_RATE, min(0.5, 1.0 - math.log10((t + 1) / 25)))


def state_to_bucket(state):
    bucket_indice = []
    for i in range(len(state)):
        if state[i] <= STATE_BOUNDS[i][0]:
            bucket_index = 0
        elif state[i] >= STATE_BOUNDS[i][1]:
            bucket_index = NUM_BUCKETS[i] - 1
        else:
            # Mapping the state bounds to the bucket array
            bound_width = STATE_BOUNDS[i][1] - STATE_BOUNDS[i][0]
            offset = (NUM_BUCKETS[i] - 1) * STATE_BOUNDS[i][0] / bound_width
            scaling = (NUM_BUCKETS[i] - 1) / bound_width
            bucket_index = int(round(scaling * state[i] - offset))
            # For easier visualization of the above, you might want to use
            # pen and paper and apply some basic algebraic manipulations.
            # If you do so, you will obtain (B-1)*[(S-MIN)]/W], where
            # B = NUM_BUCKETS, S = state, MIN = STATE_BOUNDS[i][0], and
            # W = bound_width. This simplification is very easily
            # to visualize, i.e. num_buckets x percentage in width.
        bucket_indice.append(bucket_index)
    return tuple(bucket_indice)


if __name__ == "__main__":
    print('Training ...')
    train()
