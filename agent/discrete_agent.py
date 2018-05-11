import random
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
import math

class DQNAgent():
    def __init__(self, env, gamma=0.99, 
        epsilon=1.0, epsilon_min=0.00, epsilon_log_decay=0.99, 
        alpha=0.001, alpha_decay=0.01, batch_size=128, quiet=False):
        
        self.env = env
        self.memory = []
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_log_decay
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self._batch_size = batch_size
        self.quiet = quiet
        self._state_dim = np.prod(np.array(env.observation_space.shape))
        
        self.model = Sequential()
        self.model.add(Dense(512, input_dim=self._state_dim, activation='relu'))
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(3, activation='linear'))
        self.model.compile(loss='mse', optimizer=Adam(lr=self.alpha, decay=self.alpha_decay))

    def store_step(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, step):
        epsilon = max(self.epsilon_min, min(self.epsilon, 1.0 - math.log10((step + 1) * self.epsilon_decay)))
        return int(self.env.action_space.sample()) if (np.random.random() <= epsilon) else np.argmax(self.model.predict(state))

    def train(self):
        batch_size = self._batch_size
        x_batch, y_batch = [], []
        
        np.random.shuffle(self.memory)
        
        batches = []
        for i in range(0, len(self.memory), self._batch_size):
            batches.append(self.memory[i:i + self._batch_size])
            
        for b in batches: 
            
            for state, action, reward, next_state, done in b:
                y_target = self.model.predict(state)
                y_target[0][action] = reward if done else reward + self.gamma * np.max(self.model.predict(next_state)[0])
                x_batch.append(state[0])
                y_batch.append(y_target[0])
        
            self.model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.memory = []
        