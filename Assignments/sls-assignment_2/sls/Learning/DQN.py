from keras.layers import Dense, Conv2D, Flatten
from keras.models import Sequential, clone_model
from keras.optimizers import RMSprop
from collections import deque
from keras import backend as K
import tensorflow as tf


import random
import numpy as np
from Learning.Evaluation import Evaluation


class deep_q_net:

    def __init__(self,
                 state_size,
                 action_size,
                 path="Learning/Weights/weights.h5",
                 new_weights=True,
                 memory_size=100000,
                 replay_start_size=6000,
                 epsilon=1,
                 epsilon_min=.05,
                 max_step_for_epsilon_decay=125000*3,
                 alpha=0.6,
                 beta=0.4,
                 beta_inc=0.0000005):

        self.state_size = state_size
        self.action_size = action_size
        self.path = path
        
        self.memory = deque(maxlen=memory_size)
        self.use_prio_buffer = False

        self.gamma = 0.95    # discount rate
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = 0.995
        self.max_step_for_lin_epsilon_decay = max_step_for_epsilon_decay

        self.epsilon_decay_linear = self.epsilon / self.max_step_for_lin_epsilon_decay

        self.learning_rate = 0.00025
        self.replay_start_size = replay_start_size
        self.model = self._build_model()
        self.target_model = clone_model(self.model) #self._build_model()
        self.target_model.compile(optimizer='sgd', loss='mse')

        self.step = 0

        if not new_weights:
            self.model.load_weights(path)

        self.update_target()

        self.callback = Evaluation.create_tensorboard()

    def _build_model(self):

        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.5
        
        K.tensorflow_backend.set_session(tf.compat.v1.Session(config=config))

        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Conv2D(16, 5, strides=1,
                         activation='relu',
                         input_shape=self.state_size,
                         #data_format="channels_first",
                         kernel_initializer='he_normal',
                         padding='same'))
        model.add(Conv2D(32, 3, strides=1,
                         activation='relu',
                         kernel_initializer='he_normal',
                         #data_format="channels_first"
                         padding='same'))
        # model.add(Convolution2D(64, 3, strides=1, activation='relu'))
        model.add(Flatten())
        model.add(Dense(128, activation='relu', kernel_initializer='he_normal'))
        model.add(Dense(self.action_size, activation='linear', kernel_initializer='he_normal'))

        def abs_err(prediction, target):
            return K.abs(prediction - target)

        model.compile(loss=self._huber_loss,#'mse',
                      # optimizer=Adam(lr=self.learning_rate))
                      optimizer=RMSprop(lr=self.learning_rate),
                      metrics=[abs_err])

        return model

    def _huber_loss(self, target, prediction):
        error = prediction - target
        print("Loss: ", K.mean(K.sqrt(1 + K.square(error)) - 1, axis=-1))
        return K.mean(K.sqrt(1 + K.square(error)) - 1, axis=-1)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        #state = np.moveaxis(state[0],2,0)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def get_minibatch(self, batch_size):
        if not self.use_prio_buffer:
            return None, random.sample(self.memory, batch_size), None
        else:
            self.beta += self.beta_inc
            states, actions, rewards, next_states, dones, weights, idxes = self.prio_memory.sample(batch_size, self.beta)
            minibatch = zip(states, actions, rewards, next_states, dones)
            return idxes, minibatch, weights
            # return None, self.prio_memory.sample(batch_size,self.beta), None

    def replay(self, batch_size):

        if (not self.use_prio_buffer and len(self.memory) < self.replay_start_size) or \
                (self.use_prio_buffer and len(self.prio_memory) < self.replay_start_size):
            return

        tree_idx, minibatch, is_weights = self.get_minibatch(batch_size)

        # random.sample(self.memory, batch_size)
        # except ValueError:
            # minibatch = self.memory

        if isinstance(self.state_size, int):
            state_array = np.ndarray(shape=(32, self.state_size))
        else:
            state_array = np.ndarray(shape=(32, self.state_size[0], self.state_size[1], self.state_size[2]))

        y = np.ndarray(shape=(32, self.action_size))
        actions = np.ndarray(shape=(32,), dtype=int)
        i = 0
        for state, action, reward, next_state, done in minibatch:

            # self.model.fit(state, target_f, epochs=1, verbose=0)

            state_array[i] = state
            y[i] = self.set_target(reward, state, next_state, action, done)
            actions[i] = int(action)
            i += 1
            # self.model.fit(state, target_f, epochs=1, verbose=0)

        # self.model.fit(prediction, y, batch_size=1)
        # self.model.train_on_batch(state_array, y)
        self.train(state_array, y, is_weights, tree_idx, actions, minibatch)

        self.decrease_epsilon_linear()

    def train(self, states, target, is_weights, tree_idx, actions, minibatch):
        if not self.use_prio_buffer:
            self.model.train_on_batch(states, target)
        # self.model.fit(prediction, target, verbose=0, callbacks=[self.callback])

        else:
            is_weights = np.reshape(is_weights, newshape=(is_weights.shape[0]))
            self.model.fit(states, target, sample_weight=is_weights, verbose=0)

            #abs_errs = np.abs(self.model.predict_on_batch(states)[np.arange(32), actions] - target[np.arange(32), actions])

            batch_size = len(minibatch)

            predictions = self.model.predict_on_batch(states)[np.arange(batch_size), actions]
            tar_vals = [self.set_target(r,s,na,a,d) for r,s,na,a,d in minibatch] #target[np.arange(batch_size), actions]

            # huber loss
            error = predictions - tar_vals
            #print("Loss: ", K.mean(K.sqrt(1 + K.square(error)) - 1, axis=-1))
            abs_errs = np.sqrt(1 + np.square(error)) - 1

            #abs_errs = self._huber_loss(tar_vals, predictions)

            abs_errs = abs_errs + 0.01

            self.prio_memory.update_priorities(tree_idx, abs_errs)

    def set_target(self, reward, state, next_state, action, done):
        target = reward
        if not done:
            target = reward + self.gamma * \
                     np.amax(self.target_model.predict(next_state)[0])
        target_f = self.model.predict(state)
        target_f[0][action] = target
        return target_f

    def decrease_epsilon_factor(self):
        if (not self.use_prio_buffer and len(self.memory) < self.replay_start_size) or \
                (self.use_prio_buffer and len(self.prio_memory) < self.replay_start_size):
            return
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def decrease_epsilon_linear(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay_linear

    def save_weights(self):
        self.model.save_weights(self.path)

    def update_target(self):
        self.target_model.set_weights(self.model.get_weights())
