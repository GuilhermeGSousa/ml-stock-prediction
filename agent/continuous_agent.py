import tensorflow as tf
import numpy as np

class StochasticPolicyGradientAgent():
    
    def __init__(self, env, learning_rate = 0.001, discount_rate = 0.99):
        self._optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
        self._sess = tf.Session()
        
        self._discount_rate = discount_rate
        self._state_buffer  = []
        self._reward_buffer = []
        self._action_buffer = []
        
        states_dim = np.prod(np.array(env.observation_space.shape))
        action_dim = np.prod(np.array(env.action_space.shape))
        print(action_dim)
        self._states = tf.placeholder(tf.float32, shape=(env.observation_space.shape), name="states")
        states = tf.reshape(self._states, [1,states_dim])
        
        # Building mu Model
        h1 = h2 = h3 = 128
        
        mu_hidden = tf.layers.dense(states, h1, 
                                    activation = tf.nn.relu, 
                                    name = 'dense', 
                                    kernel_initializer=tf.random_normal_initializer)
        mu_hidden_2 = tf.layers.dense(mu_hidden, h2, 
                                      activation = tf.nn.relu, 
                                      name = 'dense_1', 
                                      kernel_initializer=tf.random_normal_initializer)
        mu_hidden_3 = tf.layers.dense(mu_hidden_2, h3, 
                                      activation = tf.nn.relu, 
                                      name = 'dense_2', 
                                      kernel_initializer=tf.random_normal_initializer)
        self._mu = tf.layers.dense(mu_hidden_3, 1, 
                                   activation = tf.tanh, 
                                   name = 'mu', 
                                   kernel_initializer=tf.random_normal_initializer)
        self._mu = tf.squeeze(self._mu)
        
        # Building sigma Model
        
        sig_hidden = tf.layers.dense(states, h1, 
                                     activation = tf.nn.relu, 
                                     name = 'sigma_dense', 
                                     kernel_initializer=tf.random_normal_initializer)
        sig_hidden_2 = tf.layers.dense(sig_hidden, h2, 
                                       activation = tf.nn.relu, 
                                       name = 'sig_dense_1', 
                                       kernel_initializer=tf.random_normal_initializer)
        sig_hidden_3 = tf.layers.dense(sig_hidden_2, h3, 
                                       activation = tf.nn.relu, 
                                       name = 'sig_dense_2', 
                                       kernel_initializer=tf.random_normal_initializer)
        self._sigma = tf.layers.dense(sig_hidden_3, action_dim, 
                                      activation = tf.nn.relu, 
                                      name = 'sigma', 
                                      kernel_initializer=tf.random_normal_initializer)
        self._sigma = tf.nn.softplus(self._sigma) + 1e-5
        self._sigma = tf.squeeze(self._sigma)
        
        self._normal_dist = tf.contrib.distributions.Normal(self._mu, self._sigma)
        
        self.action = self._normal_dist._sample_n(1)
        self.action = tf.clip_by_value(self.action, env.action_space.low[0], env.action_space.high[0])
        
        #Computing loss function
        
        self._discounted_rewards = tf.placeholder(tf.float32, (None, 1), name="discounted_rewards")
        self._taken_actions = tf.placeholder(tf.float32, (None, 1), name="taken_actions")
        
        #Is this reward function correct?
        self._loss = -self._normal_dist.log_prob(self._taken_actions) * self._discounted_rewards
        
        self._train_op = self._optimizer.minimize(self._loss)        
        
        
        self._sess.run(tf.global_variables_initializer())
                
    def act(self, state):
        action = self._sess.run(self.action, feed_dict={
            self._states: state})
        return action
    
    def train(self):
        
        #After applying gradients
        self._state_buffer  = []
        self._reward_buffer = []
        self._action_buffer = []
    
    def store_step(self, action, state, reward):
        self._state_buffer.append(state)
        self._reward_buffer.append(reward)
        self._action_buffer.append(action)
        
    def _discount_rewards(self):
        r = 0
        N = len(self._reward_buffer)
        discounted_rewards = np.zeros(N)
        for t in reversed(range(N)):
            r = r + self._reward_buffer[t] * self._discount_rate
            discounted_rewards[t] = r