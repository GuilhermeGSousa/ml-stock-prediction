import tensorflow as tf
import numpy as np

class StochasticPolicyGradientAgent():
    """
    A Gaussian Policy Gradient based agent implementation
    """
    def __init__(self, env, learning_rate = None, discount_rate = 0.99, batch_size = 1):
        if learning_rate is None:
            self._optimizer = tf.train.AdamOptimizer()
        else:
            self._optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
        self._sess = tf.Session()
        self._env = env
        self._batch_size = batch_size
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
        h1 = 512
        h2 = 256
        h3 = 128
        h4 = 64
        
        mu_hidden = tf.layers.dense(self._states, h1, 
                                    activation = tf.nn.tanh, 
                                    name = 'dense_0', 
                                    kernel_initializer=init)
        mu_hidden_2 = tf.layers.dense(mu_hidden, h2, 
                                      activation = tf.nn.tanh, 
                                      name = 'dense_1', 
                                      kernel_initializer=init)
        mu_hidden_3 = tf.layers.dense(mu_hidden_2, h3, 
                                      activation = tf.nn.tanh, 
                                      name = 'dense_2', 
                                      kernel_initializer=init)
        mu_hidden_4 = tf.layers.dense(mu_hidden_3, h4, 
                                      activation = tf.nn.tanh, 
                                      name = 'dense_3', 
                                      kernel_initializer=init)
        self._mu = tf.layers.dense(mu_hidden_3, 1,
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
        self._sigma = tf.layers.dense(sig_hidden_3, 1, 
                                      activation = tf.exp, 
                                      name = 'sigma', 
                                      kernel_initializer=init)
        self._sigma = tf.squeeze(self._sigma)
        self._sigma = tf.add(self._sigma, self._sigma)
        
        #Sampling action from distribuition
        
        self._normal_dist = tf.contrib.distributions.Normal(self._mu, 0.5)
        self._action = self._normal_dist.sample()
        
        #Computing loss function
        
        self._discounted_rewards = tf.placeholder(tf.float32, (None, 1), name="discounted_rewards")
        self._taken_actions = tf.placeholder(tf.float32, (None, 1), name="taken_actions")
        
        self._loss = -tf.reduce_mean(tf.log(1e-5 + self._normal_dist.prob(self._taken_actions)) * self._discounted_rewards,0)
         
            
        #gvs = self._optimizer.compute_gradients(self._loss)
        #capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
        #self._train_op = self._optimizer.apply_gradients(capped_gvs)
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
        rewards = [[r] for r in rewards]
        samples = []
        for t in range(len(self._state_buffer)-1):
            samples.append([self._state_buffer[t], rewards[t], self._action_buffer[t]])
            
        np.random.shuffle(samples)
        batches = []
        
        for i in range(0, len(samples), self._batch_size):
            batches.append(samples[i:i + self._batch_size])
            
        for b in range(len(batches)):
            batch = batches[b]
            states_batch = [row[0] for row in batch]
            actions_batch = [row[2] for row in batch]
            rewards_batch = [row[1] for row in batch]
            
            feed_dict={
                self._states: states_batch,
                self._discounted_rewards: rewards_batch,
                self._taken_actions: actions_batch}

            self._sess.run([self._train_op], feed_dict=feed_dict)
        
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
