

from keras import Sequential
from keras import losses,optimizers
from keras.layers import Dense
import gym
import random
import numpy as np
import tensorflow as tf
from keras import backend as K


class ReplayMemory():

    def __init__(self,memory_size):

        self.memory_size = memory_size
        self.memory = []

    def remember(self,what_to_remember):
        if not isinstance(what_to_remember,tuple):
            print ("Error")

        else:
            self.memory.append(what_to_remember)

        if len(self.memory)>self.memory_size:
            self.memory=[]

    def sample_minibatch(self,batch_size):

        mini_batch = random.sample(self.memory,batch_size)

        return mini_batch






class DQNAgent():

    def __init__(self, state_size, action_size, epsilon,gamma,size_batch):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = epsilon
        self.min_epsilon=0.01
        self.gamma = gamma
        self.epsilon_decay =0.99
        self.minibatch_size = size_batch
        self.model_network = self.build_network()
        self.target_network = self.build_network()
        self.copy_paramters()


        self.memory = ReplayMemory(2000)

    def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond  = K.abs(error) <= clip_delta

        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)

        return K.mean(tf.where(cond, squared_loss, quadratic_loss))



    def build_network(self):

        Model = Sequential()
        Model.add(Dense(24,input_dim = self.state_size,activation="relu"))
        Model.add(Dense(24, activation="relu"))
        Model.add(Dense(self.action_size,activation="linear"))

        Model.compile(loss=self._huber_loss, optimizer=optimizers.Adam(lr=0.01))

        return Model


    def act(self,state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model_network.predict(state)

        return np.argmax(act_values[0])  # returns action

        #def copy_paramters():

            #take model_network params replace to target


    def replay(self):

        batch = self.memory.sample_minibatch(self.minibatch_size)

        states, targets_f = [], []

        for s,action,reward,next_state,done in batch:

            target = self.model_network.predict(s)
            if done:
                target[0][action] = reward
            else:
                target[0][action] = reward + self.gamma * np.amax(self.target_network.predict(next_state)[0])


            self.model_network.fit(state,target, epochs=1, verbose=0)


        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay



    def copy_paramters(self):

        self.target_network.set_weights(self.model_network.get_weights())




if __name__ == "__main__":
    # initialize gym environment and the agent
    env = gym.make('CartPole-v0')

    state_size = env.observation_space.shape[0]
    action_space =env.action_space.n

    agent = DQNAgent(state_size,action_space, 1, 0.95, 50)
    episodes=1000
    done=False
    # Iterate the game
    for e in range(episodes):
        # reset state in the beginning of each game
        state = env.reset()

        state = np.reshape(state, [1, 4])
        # time_t represents each frame of the game
        # Our goal is to keep the pole upright as long as possible until score of 500
        # the more time_t the more score
        for time_t in range(500):
            # turn this on if you want to render
            env.render()
            # Decide action
            action = agent.act(state)
            # Advance the game to the next frame based on the action.
            # Reward is 1 for every frame the pole survived
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10

            next_state = np.reshape(next_state, [1, 4])
            # Remember the previous state, action, reward, and done
            agent.memory.remember((state, action, reward, next_state, done))
            state = next_state
            if done:
                agent.copy_paramters()
                # print the score and break out of the loop
                print("episode: {}/{}, score: {}"
                      .format(e, episodes, time_t))
                break

            if len(agent.memory.memory) > agent.minibatch_size:

                agent.replay()
            #if time_t % 100 == 0:








