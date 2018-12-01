import sys
import gym
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import  adam
import numpy as np
from keras import losses






class ValueEstimator:

    def __init__(self, state_size):
        self.state_size = state_size
        self.model = self.build_value_estimator()


    def build_value_estimator(self):
        model = Sequential()

        model.add(Dense(24, input_dim= self.state_size, activation='relu', kernel_initializer='glorot_uniform'))
        model.add(Dense(1, activation=None, kernel_initializer='glorot_uniform'))

        model.compile(optimizer=adam(lr=0.001), loss='mean_squared_error')

        return model

    def train_value_estimator(self, target):

        self.model.fit(self.state, target, epochs=1, verbose=0)

    def discounted_rewards(self, rewards):

        running_add =0

        discounted_rewards = np.zeros_like(rewards)
        for t in reversed(range(0,len(rewards))):
            running_add=running_add*0.99 + rewards[t]
            discounted_rewards[t]=running_add

        return discounted_rewards



    def train_model_valuestimator(self,agent):

        episode_length = len(agent.states)

        obs = np.zeros((episode_length, self.state_size))
        advantage = np.zeros((episode_length, agent.action_size))
        baseline_value=np.zeros(episode_length)

        discounted_rewards= self.discounted_rewards(agent.rewards)
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)

        for i in range(episode_length):
            obs[i] = agent.states[i]
            baseline_value[i] = self.model.predict(obs[i, :].reshape(1,4),batch_size=1)
            advantage[i][agent.actions[i]] = discounted_rewards[i] - np.sum(baseline_value)


        self.model.fit(obs, self.discounted_rewards(agent.rewards),epochs=1,verbose=0)

        print (self.model.layers[1].get_weights()[0])

        return (baseline_value)





class ReinforceAgent:

    def __init__(self,state_size, action_size,hidden_1,hidden_2):

        self.render = True
        self.state_size= state_size
        self.action_size = action_size
        self.hidden_1 = hidden_1
        self.hidden_2 = hidden_2

        self.model = self.build_model()

        self.states, self.actions, self.rewards = [],[],[]




    def build_model(self):

        model = Sequential()

        model.add(Dense(self.hidden_1, input_dim=self.state_size,activation='relu', kernel_initializer='glorot_uniform'))
        model.add(Dense(self.hidden_2, activation='relu', kernel_initializer='glorot_uniform'))

        model.add(Dense(self.action_size, activation='softmax', kernel_initializer='glorot_uniform'))

        model.compile(optimizer=adam(lr=0.001), loss='categorical_crossentropy')
        return model

    def append_sample(self, state, action, reward):
        self.states.append(state)
        self.rewards.append(reward)
        self.actions.append(action)

    def get_action(self,state):

        policy = self.model.predict(state,batch_size=1).flatten()

        return np.random.choice(self.action_size,1,p=policy)[0]


    def discounted_rewards(self,rewards):

        running_add =0

        discounted_rewards = np.zeros_like(rewards)
        for t in reversed(range(0,len(rewards))):
            running_add = running_add*0.99 + rewards[t]
            discounted_rewards[t] = running_add

        return discounted_rewards


    def train_model(self,baseline_values_vec):

        episode_length = len(self.states)

        obs = np.zeros((episode_length, self.state_size))
        advantage = np.zeros((episode_length, self.action_size))

        discounted_rewards = self.discounted_rewards(self.rewards)
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)

        for i in range(episode_length):
            baseline_value = baseline_values_vec[i]
            advantage[i][self.actions[i]] = discounted_rewards[i] - np.sum(baseline_value)
            print (baseline_value)
            advantage[i][self.actions[i]]

        self.model.fit( obs, advantage,epochs=1,verbose=0)

        self.states, self.actions, self.rewards = [], [], []





if __name__ == "__main__":
    # In case of CartPole-v1, you can play until 500 time step
    env = gym.make('CartPole-v1')
    # get size of state and action from environment
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # make REINFORCE agent
    agent = ReinforceAgent(state_size, action_size,24,24)
    value_estimator = ValueEstimator(state_size)
    EPISODES = 100

    scores, episodes = [], []

    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])



        while not done:
            if agent.render:
                env.render()

            # get action for the current state and go one step in environment
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            reward = reward if not done or score == 499 else -100

            # save the sample <s, a, r> to the memory
            agent.append_sample(state, action, reward)

            score += reward
            state = next_state

            if done:
                # every episode, agent learns from sample returns

                bas_val = value_estimator.train_model_valuestimator(agent)

                agent.train_model(bas_val)


                # every episode, plot the play time
                score = score if score == 500 else score + 100
                scores.append(score)
                episodes.append(e)
                print("episode:", e, "  score:", score)
                if np.mean(scores[-min(10, len(scores)):]) > 490:
                    sys.exit()

