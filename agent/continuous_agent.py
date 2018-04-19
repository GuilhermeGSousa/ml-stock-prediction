import tensorflow as tf
import numpy as np

class StochasticPolicyGradientAgent():
    """
    A Gaussian Policy Gradient based agent implementation
    """
    def __init__(self, env, learning_rate = None, discount_rate = 0.99):
        if learning_rate is None:
            self._optimizer = tf.train.AdamOptimizer()
        else:
            self._optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
        self._sess = tf.Session()
        self._env = env
        
        self._discount_rate = discount_rate
        self._state_buffer  = []
        self._reward_buffer = []
        self._action_buffer = []
        
        
        state_dim = np.prod(np.array(env.observation_space.shape))

        self._states = tf.placeholder(tf.float32, 
                                      shape=(None, state_dim), 
                                      name="states")
        

        init = tf.contrib.layers.xavier_initializer()
        
        # neural featurizer parameters
        h1 = 256
        h2 = 256
        h3 = 128
        h4 = 128
        
        mu_hidden = tf.layers.dense(self._states, h1, 
                                    activation = tf.sigmoid, 
                                    name = 'dense_0', 
                                    kernel_initializer=init)
        mu_hidden_2 = tf.layers.dense(mu_hidden, h2, 
                                      activation = tf.sigmoid, 
                                      name = 'dense_1', 
                                      kernel_initializer=init)
        mu_hidden_3 = tf.layers.dense(mu_hidden_2, h3, 
                                      activation = tf.sigmoid, 
                                      name = 'dense_2', 
                                      kernel_initializer=init)
        mu_hidden_4 = tf.layers.dense(mu_hidden_3, h4, 
                                      activation = tf.sigmoid, 
                                      name = 'dense_3', 
                                      kernel_initializer=init)
        self._mu = tf.layers.dense(mu_hidden_4, 1,
                                   activation = tf.tanh,
                                   name = 'mu', 
                                   kernel_initializer=init)
        self._mu = tf.squeeze(self._mu)
        

        # Building sigma Model
        
        sig_hidden = tf.layers.dense(self._states, h1, 
                                     activation = tf.sigmoid, 
                                     name = 'sigma_dense_0', 
                                     kernel_initializer=init)
        sig_hidden_2 = tf.layers.dense(sig_hidden, h2, 
                                       activation = tf.sigmoid, 
                                       name = 'sig_dense_1', 
                                       kernel_initializer=init)
        sig_hidden_3 = tf.layers.dense(sig_hidden_2, h3, 
                                       activation = tf.sigmoid, 
                                       name = 'sig_dense_2', 
                                       kernel_initializer=init)
        sig_hidden_4 = tf.layers.dense(sig_hidden_3, h4, 
                                       activation = tf.sigmoid, 
                                       name = 'sig_dense_3', 
                                       kernel_initializer=init)
        self._sigma = tf.layers.dense(sig_hidden_4, 1, 
                                      activation = tf.exp, 
                                      name = 'sigma', 
                                      kernel_initializer=init)
        self._sigma = tf.squeeze(self._sigma)
        self._sigma = tf.add(self._sigma, 1e-5)
        
        #Sampling action from distribuition
        
        self._normal_dist = tf.contrib.distributions.Normal(self._mu, self._sigma)
        self._action = self._normal_dist.sample()
        
        #Computing loss function
        
        self._discounted_rewards = tf.placeholder(tf.float32, (None, 1), name="discounted_rewards")
        self._taken_actions = tf.placeholder(tf.float32, (None, 1), name="taken_actions")
        
        self._loss = -tf.reduce_mean(tf.log(1e-10 + self._normal_dist.prob(self._taken_actions)) * self._discounted_rewards,0)
                                                            
        self._train_op = self._optimizer.minimize(self._loss)        
        
        self._sess.run(tf.global_variables_initializer())
                
    def act(self, state):
        mu, sigma, action = self._sess.run([self._mu, self._sigma, self._action], feed_dict={
            self._states: state})
        action = np.clip(action, self._env.action_space.low[0], self._env.action_space.high[0])
        #print("Sigma: {}, Mu: {}, Action: {}".format(sigma, mu, action))
        
        return action
    
    def train(self): 
        rewards = self._discount_rewards().tolist()
        norm_rewards = rewards
        norm_rewards -= np.mean(rewards)
        #norm_rewards /= np.std(rewards)
        rewards = [[r] for r in rewards]
        norm_rewards = [[r] for r in norm_rewards]
        feed_dict={
            self._states: self._state_buffer,
            self._discounted_rewards: norm_rewards,
            self._taken_actions: self._action_buffer}
        
        self._sess.run([self._train_op], feed_dict=feed_dict)
        #print(self._sess.run(self._loss, feed_dict=feed_dict))
        
        #After applying gradients
        self._state_buffer  = []
        self._reward_buffer = []
        self._action_buffer = []
    
    def store_step(self, action, state, reward):
        self._state_buffer.append(state)
        self._reward_buffer.append(np.array(reward))
        self._action_buffer.append(np.array([action]))
        
    def _discount_rewards(self):
        r = 0
        N = len(self._reward_buffer)
        discounted_rewards = np.zeros(N)
        for t in reversed(range(N)):
            r = r + self._reward_buffer[t] * self._discount_rate
            discounted_rewards[t] = r
        return discounted_rewards
