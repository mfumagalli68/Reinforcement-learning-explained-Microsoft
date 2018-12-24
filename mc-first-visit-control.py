import gym
import numpy as np
from collections import defaultdict
from envclass import *
import sys


def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.

    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.

    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """

    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A

    return policy_fn


def mc_control(env,no_episodes,discountfactor=1):

    epsilon=0.1
    returnsum = defaultdict(float)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    returns_count = defaultdict(float)
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    for i in range(1,no_episodes):
        if i % 1000 == 0:
            print("\rEpisode {}/{}.".format(i, no_episodes), end="")
            sys.stdout.flush()

        episode = []
        state = env.reset()
        for t in range(100):

            probs = policy(state)
            action = np.random.choice(np.arange(len(probs)), p=probs)
            next_state, reward, done, _ = env.step(action)

            episode.append((next_state, action, reward))
            if done:
                break

            state = next_state
        print(episode)
        visited_states = [i[0] for i in episode]
        visited_action = [i[1] for i in episode]

        for z in zip(visited_states,visited_action):

            # find G = Find first occurence of state action pair and sum rewards from there to the end.
            first_occurence = [i for i,x in enumerate(episode) if x[0] == state and x[1] == action ]
            if len(first_occurence)>0:
                first_occurence= first_occurence[0]
                G = sum([x[2] * (discountfactor ** i) for i, x in enumerate(episode[first_occurence:])])

                # Update Q
                returnsum[z] += G
                returns_count[z] += 1.0
                Q[z[0]][z[1]] = Q[z[0]][z[1]] + ((returnsum[z] - Q[z[0]][z[1]]) / returns_count[z])
            else:
                break


        epsilon = epsilon*i


    return Q,policy


env = BlackjackEnv()


# For plotting: Create value function from action-value function
# by picking the best action at each state
Q , policy = mc_control(env,200000)
V = defaultdict(float)
for state, actions in Q.items():
    action_value = np.max(actions)
    V[state] = action_value

plot_value_function(V, title="Optimal Value Function")







