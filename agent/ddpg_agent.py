import tensorflow as tf
import numpy as np
import random

class DDPGAgent():
    def __init__(self, env, learning_rate = 0.001, discount_rate = 0.99, batch_size = 128, quiet = True):
        self.actor = Actor(env, learning_rate = learning_rate, quiet = quiet)
        self.critic = Critic(env, learning_rate = learning_rate, quiet = quiet)
        
        self._batch_size = batch_size
        self._discount_rate = discount_rate
        # Memory
        self._state_buffer  = []
        self._action_buffer = []
        self._q_buffer = []
        
    
    def store_step(self, state, action, reward, next_state):
        self._state_buffer.append(state)
        self._action_buffer.append(np.array([action]))
        
        next_action = self.actor.act([next_state])
        q_next = self.critic.predict_q([next_state], [[next_action]])
        q_expected = reward + self._discount_rate * q_next
        self._q_buffer.append(np.array(q_expected))
    
    def train(self):
        self._q_buffer = [[q] for q in self._q_buffer]
        samples = []
        for t in range(len(self._state_buffer)):
            samples.append([self._state_buffer[t], self._action_buffer[t], self._q_buffer[t]])
        np.random.shuffle(samples)
        batches = []
        
        for i in range(0, len(samples), self._batch_size):
            batches.append(samples[i:i + self._batch_size])
            
        for batch in batches:
            states_batch = [row[0] for row in batch]
            actions_batch = [row[1] for row in batch]
            q_batch = [row[2] for row in batch]
            
            self.critic.train(states_batch, q_batch, actions_batch)
            action_grads_batch = self.critic.get_action_grads(states_batch, actions_batch)
            action_grads_batch = [[a] for a in action_grads_batch]
            self.actor.train(states_batch, action_grads_batch)
        
        #After applying gradients
        self._state_buffer  = []
        self._action_buffer = []
        self._q_buffer = []
            
class Actor():

    def __init__(self, env, learning_rate = 0.001, quiet = True):
        
        self._optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
        self._sess = tf.Session()
        self._env = env
        self._quiet = quiet
        
        state_dim = np.prod(np.array(env.observation_space.shape))

        self._state = tf.placeholder(tf.float32, 
                                      shape=(None, state_dim), 
                                      name="states")
 
        
        init = tf.contrib.layers.xavier_initializer()
        
        # neural featurizer parameters
        h1 = 256
        h2 = 128
        h3 = 128
        
        action_hidden = tf.layers.dense(self._state, h1, 
                                    activation = tf.nn.tanh, 
                                    name = 'dense_0', 
                                    kernel_initializer=init)
        action_hidden_2 = tf.layers.dense(action_hidden, h2, 
                                      activation = tf.nn.tanh, 
                                      name = 'dense_1', 
                                      kernel_initializer=init)
        action_hidden_3 = tf.layers.dense(action_hidden_2, h3, 
                                      activation = tf.nn.tanh, 
                                      name = 'dense_2', 
                                      kernel_initializer=init)
        self._action = tf.layers.dense(action_hidden_3, 1,
                                   activation = tf.tanh,
                                   name = 'action', 
                                   kernel_initializer=init)
        self._action = tf.squeeze(self._action)
        
        #Computing training op
        
        self._trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
         
        self._action_gradients = tf.placeholder(tf.float32, [None, 1], name="action_grad")
        
        self._var_grads = tf.gradients(self._action,self._trainable_vars,-self._action_gradients)
        
        self._train_op = self._optimizer.apply_gradients(zip(self._var_grads,self._trainable_vars))      
        
        self._sess.run(tf.global_variables_initializer())
                
    def act(self, state):
        
        action = self._sess.run(self._action, feed_dict={
            self._state: state})

        if not self._quiet:
            print("Action: {}".format(action))
        
        return action
    
    def train(self, states_batch, actions_grads_batch): 
        feed_dict={
            self._state: states_batch,
            self._action_gradients: actions_grads_batch}
        self._sess.run([self._train_op], feed_dict=feed_dict)


class Critic():

    def __init__(self, env, learning_rate = 0.001, quiet = True):
        
        self._optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
        self._sess = tf.Session()
        self._env = env
        self._quiet = quiet
           
        state_dim = np.prod(np.array(env.observation_space.shape))

        self._state = tf.placeholder(tf.float32, 
                                      shape=(None, state_dim), 
                                      name="states")

        self._action = tf.placeholder(tf.float32,shape=[None,1], name="action")
            
        init = tf.contrib.layers.xavier_initializer()
        
        # neural featurizer parameters
        h1 = 256
        h2 = 128
        h3 = 128
        
        
        q_hidden = tf.layers.dense(tf.concat([self._state, self._action], 1), h1, 
                                    activation = tf.nn.tanh, 
                                    name = 'q_dense_0', 
                                    kernel_initializer=init)
        q_hidden_2 = tf.layers.dense(q_hidden, h2, 
                                      activation = tf.nn.tanh, 
                                      name = 'q_dense_1', 
                                      kernel_initializer=init)
        q_hidden_3 = tf.layers.dense(q_hidden_2, h3, 
                                      activation = tf.nn.tanh, 
                                      name = 'q_dense_2', 
                                      kernel_initializer=init)
        self._q = tf.layers.dense(q_hidden_3, 1,
                                   activation = None,
                                   name = 'q', 
                                   kernel_initializer=init)
        
        self._q = tf.squeeze(self._q)
        
        #Computing training op
        
        self._trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        
        self._q_expected = tf.placeholder(tf.float32,shape=[None,1], name="q_expected")
        
        self._loss = tf.losses.mean_squared_error(self._q_expected,self._q)
        
        self._train_op = self._optimizer.minimize(self._loss)
        
        self._action_gradients = tf.squeeze(tf.gradients(self._q ,self._action))
                
        self._sess.run(tf.global_variables_initializer())
                
    def predict_q(self, state, action):
        
        q = self._sess.run(self._q, feed_dict={
            self._state: state,
            self._action: action})
        return q
    
    def get_action_grads(self, state, action):
        grads = self._sess.run(self._action_gradients, feed_dict={
            self._state: state,
            self._action: action})
        return grads
    
    def train(self, states_batch, q_batch, actions_batch): 
        feed_dict={
            self._state: states_batch,
            self._q_expected: q_batch,
            self._action: actions_batch}

        self._sess.run([self._train_op], feed_dict=feed_dict)
        
        