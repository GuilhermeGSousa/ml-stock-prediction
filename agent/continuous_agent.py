import tensorflow as tf
import numpy as np

class StochasticPolicyGradientAgent():
    """
    A Gaussian Policy Gradient based on lantunes's agent for MountainCarContinuous
    github.com/lantunes/mountain-car-continuous
    """
    def __init__(self, env, learning_rate = 0.001, discount_rate = 0.99):
        self._optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
        self._sess = tf.Session()
        self._env = env
        
        self._discount_rate = discount_rate
        self._state_buffer  = []
        self._reward_buffer = []
        self._action_buffer = []
        
        self._phi_hidden = 128
        self._sigma_hidden = 32
        
        state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
        
        self._states = tf.placeholder(tf.float32, 
                                      shape=(None, env.observation_space.shape[0], env.observation_space.shape[1]), 
                                      name="states")
        
        state = tf.layers.flatten(self._states)
        init = tf.contrib.layers.xavier_initializer()
    # policy parameters
        self._mu_theta = tf.get_variable("mu_theta", [self._phi_hidden, 1],
                                         initializer=init)
        self._sigma_theta = tf.get_variable("sigma_theta", [self._sigma_hidden],
                                                initializer=init)
        
        # neural featurizer parameters
        self._W1 = tf.get_variable("W1", [state_dim, self._phi_hidden],
                                   initializer=init)
        self._b1 = tf.get_variable("b1", [self._phi_hidden],
                                   initializer=tf.constant_initializer(0))
        self._h1 = tf.nn.tanh(tf.matmul(state, self._W1) + self._b1)
        self._W2 = tf.get_variable("W2", [self._phi_hidden, self._phi_hidden],
                                   initializer=init)
        self._b2 = tf.get_variable("b2", [self._phi_hidden],
                                   initializer=tf.constant_initializer(0))
        self._phi = tf.nn.tanh(tf.matmul(self._h1, self._W2) + self._b2)
        
        self._mu = tf.tanh(tf.matmul(self._phi, self._mu_theta))
        
        self._sigma = tf.reduce_sum(self._sigma_theta)
        self._sigma = tf.exp(self._sigma)
            
        #Computing loss function
        
        self._discounted_rewards = tf.placeholder(tf.float32, (None, 1), name="discounted_rewards")
        self._taken_actions = tf.placeholder(tf.float32, (None, 1), name="taken_actions")
        
        self._loss = -tf.log(1e-5 + tf.sqrt(1/(2 * np.pi * self._sigma**2)) 
                             * tf.exp(-(self._taken_actions - self._mu)**2/(2 * self._sigma**2))) * self._discounted_rewards
        
        self._train_op = self._optimizer.minimize(self._loss)        
        
        self._sess.run(tf.global_variables_initializer())
                
    def act(self, state):
        mu, sigma = self._sess.run([self._mu, self._sigma], feed_dict={
            self._states: state})
        action = np.random.normal(mu, sigma)
        action = np.clip(action, self._env.action_space.low[0], self._env.action_space.high[0])
        print("Sigma: {}, Mu: {}, Action: {}".format(sigma, mu, action))
        return action[0]
    
    def train(self): 
        rewards = []
        rewards.append(self._discount_rewards().tolist())
        rewards = [[r] for r in rewards[0]]
        norm_rewards = rewards
        norm_rewards -= np.mean(rewards)
        norm_rewards /= np.std(rewards)
        feed_dict={
            self._states: self._state_buffer,
            self._discounted_rewards: rewards,
            self._taken_actions: self._action_buffer}
        
        loss = self._sess.run(self._loss, feed_dict=feed_dict)
        
        self._sess.run([self._train_op], feed_dict=feed_dict)
        
        print(loss)
        
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
        return discounted_rewards
