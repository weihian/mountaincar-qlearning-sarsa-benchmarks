import math
import random
import time

import gym
import numpy as np

# Initialize the "Cart-Pole" environment
# env = gym.make('MountainCar-v0', render_mode = "human")
env = gym.make('MountainCar-v0')

# Initializing the random number generator
np.random.seed(int(time.time()))

# Defining the environment related constants

# Number of discrete states (bucket) per state dimension
NUM_BUCKETS = (19, 15)  # (x, x', theta, theta')
# Number of discrete actions
NUM_ACTIONS = 3  # (left, right)
# Bounds for each discrete state
STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))
STATE_BOUNDS[0] = (-1.2, 0.6, 19)
STATE_BOUNDS[1] = (-0.07, 0.07, 15)
# Index of the action
ACTION_INDEX = len(NUM_BUCKETS)

# Creating a Q-Table for each state-action pair
q_table = np.zeros(NUM_BUCKETS + (NUM_ACTIONS,))

# Learning related constants
# Continue here

# Defining the simulation related constants
NUM_TRAIN_EPISODES = 1000  # 1000
MAX_TRAIN_T = 200
STREAK_TO_END = 1200
SOLVED_T = 199
VERBOSE = False


def train(discount_factor):
    # Instantiating the learning related parameters
    learning_rate = get_learning_rate(0)
    explore_rate = get_explore_rate(0)
    discount_factor = discount_factor
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
            action = select_action(state_0, explore_rate)

            # Execute the action
            obv_1, reward, done, _, _ = env.step(action)

            # Observe the result
            state_1 = state_to_bucket(obv_1)

            # kinetic energy reward function 1
            # reward = reward + 0.5 * ((abs(obv_1[1])) * (abs(obv_1[1])))

            # kinetic energy reward function 2
            # reward = reward + 1000 * ((abs(obv_1[1])) * (abs(obv_1[1])))

            # kinetic energy reward function 3
            # reward = reward + 500 * ((abs(obv_1[1])) * (abs(obv_1[1])))

            # kinetic energy reward function 4
            # reward = reward + 10 * abs(obv_1[1])

            # kinetic energy reward function 5
            # reward = reward + 20 * abs(obv_1[1])

            # kinetic energy reward function 6
            # reward = reward + 1 * (0.9 * abs(obv_1[1]) - abs(obv_0[1]))

            # potential energy reward function 1
            # reward = reward + 0.1 * abs(obv_1[0])

            # potential energy reward function 2
            # reward = reward + 20 * (abs(obv_1[0] - obv_0[0]))

            # kinetic energy and potential energy reward function
            reward = reward + ((20 * abs(obv_1[1])) + (20 * (abs(obv_1[0] - obv_0[0])))) / 2

            # Update the Q based on the result
            best_q = np.amax(q_table[state_1])
            q_table[state_0 + (action,)] += learning_rate * (
                    reward + discount_factor * best_q - q_table[state_0 + (action,)])

            # Setting up for the next iteration
            state_0 = state_1
            obv_0 = obv_1

            # Print data
            if VERBOSE:
                print("\nEpisode = %d" % episode)
                print("t = %d" % t)
                print("Action: %d" % action)
                print("State: %s" % str(state_1))
                print("Reward: %f" % reward)
                print("Best Q: %f" % best_q)
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

        # It's considered done when it's solved over 120 times consecutively
        if num_train_streaks > STREAK_TO_END:
            break

        # Update parameters
        explore_rate = get_explore_rate(episode)
        learning_rate = get_learning_rate(episode)
    print("times of win")
    print(win)
    return win


def select_action(state, explore_rate):
    # Select a random action
    if random.random() < explore_rate:
        action = random.randint(0, 2)
    # Select the action with the highest q
    else:
        action = np.argmax(q_table[state])
    return action


def set_explore_rate(t):
    global min_explore_rate
    min_explore_rate = t


def set_learning_rate(t):
    global min_learning_rate
    min_learning_rate = t


def get_explore_rate(t):
    global min_explore_rate
    return max(min_explore_rate, min(1, 1.0 - math.log10((t + 1) / 25)))


def get_learning_rate(t):
    global min_learning_rate
    return max(min_learning_rate, min(0.5, 1.0 - math.log10((t + 1) / 25)))


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


best = 0
populations = ([[random.randint(1, 100) / 100 for x in range(3)] for i in range(20)])
print(type(populations))
parents = []
new_populations = []
print(populations)


def fitness_score():
    global populations, best
    fit_value = []
    for i in range(len(populations)):
        print("pop")
        print(populations[i])
        set_learning_rate(populations[i][0])
        set_explore_rate(populations[i][1])
        fitness_value = train(populations[i][2])
        print(fitness_value)
        fit_value.append(fitness_value)
    print(fit_value)
    fit_value, populations = zip(*sorted(zip(fit_value, populations), reverse=True))
    best = fit_value[0]
    print("best")
    print(best)


def selectparent():
    global parents
    parents = populations[0:2]
    print("select parents")
    print(type(parents))
    print(parents)


def crossover():
    global parents

    cross_point = random.randint(0, 2)
    parents = parents + tuple([(parents[0][0:cross_point + 1] + parents[1][cross_point + 1:6])])
    parents = parents + tuple([(parents[1][0:cross_point + 1] + parents[0][cross_point + 1:6])])
    print("crossover")
    print(parents)


def mutation():
    global populations, parents
    if populations[1] == populations[2]:
        mute = 20
    else:
        mute = random.randint(0, 49)
    if mute == 20:
        x = random.randint(0, 2)
        y = random.randint(0, 2)
        parents[x][y] = 1-parents[x][y]
    populations = parents
    print("mutation")
    print(populations)


def genetic_algorithm():
    for i in range(10):
        fitness_score()
        selectparent()
        crossover()
        mutation()
    print("best score :")
    print(best)
    print("sequence........")
    print(populations[0])
