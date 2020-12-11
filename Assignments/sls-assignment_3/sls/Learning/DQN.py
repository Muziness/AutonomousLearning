from keras.layers import Dense, Conv2D, Flatten, Input, Lambda
from keras.models import Sequential, Model, clone_model
from keras.optimizers import RMSprop
from collections import deque
from keras import backend as K
from keras.utils import plot_model
import tensorflow as tf

from baselines.deepq.replay_buffer import PrioritizedReplayBuffer

import random
import numpy as np
import Learning.PrioritizedReplayMemory as prm
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
                 prioritized_replay=False,
                 alpha=0.6,
                 beta=0.4,
                 beta_inc=0.0000005):

        self.state_size = state_size
        self.action_size = action_size
        self.path = path

        self.use_prio_buffer = prioritized_replay
        if not prioritized_replay:
            self.memory = deque(maxlen=memory_size)
        else:
            self.prio_memory = PrioritizedReplayBuffer(memory_size, alpha)
            self.beta = beta
            self.beta_inc = beta_inc
            # self.beta_schedule = LinearSchedule(max_step_for_epsilon_decay,
            #                                     1,
            #                                     0.4)

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

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.5

        K.tensorflow_backend.set_session(tf.Session(config=config))

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
        if not self.use_prio_buffer:
            self.memory.append((state, action, reward, next_state, done))
        else:
            self.prio_memory.add(state, action, reward, next_state, done)
        # print(state, action, reward, next_state, done)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
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


class DoubleDQN(deep_q_net):

    def set_target(self, reward, state, next_state, action, done):
        target = reward
        if not done:

            next_action = np.argmax(self.model.predict(next_state))
            q_vals_next_state = self.target_model.predict(next_state)
            target = reward + self.gamma * q_vals_next_state[0][next_action]

            # target = reward + self.gamma * \
                     # self.fixed_model.predict(next_state)[0][np.argmax(self.model.predict(next_state))]
        target_f = self.model.predict(state)
        target_f[0][action] = target
        return target_f


class DDQN(deep_q_net):

    def _build_model(self):
        # Neural Net for Deep-Q learning Model

        input = Input(shape=self.state_size)

        h1 = Conv2D(16, 5, strides=1,
                    activation='relu',
                    kernel_initializer='he_normal',
                    #data_format="channels_first",
                    padding='same')(input)
        h2 = Conv2D(32, 3, strides=1,
                    activation='relu',
                    kernel_initializer='he_normal',
                    #data_format="channels_first",
                    padding='same')(h1)

        f1 = Flatten()(h2)

        # # Value Function
        # fc_v = Dense(256, activation='relu', kernel_initializer='he_normal')(f1)
        # v = Dense(1, kernel_initializer='he_normal')(fc_v)
        #
        # # Advantage Function
        # fc_a = Dense(256, activation='relu', kernel_initializer='he_normal')(f1)
        # a = Dense(self.action_size, kernel_initializer='he_normal')(fc_a)
        #
        # advt = Lambda(lambda a: a - K.mean(a, keepdims=True), output_shape=(self.action_size, ))(a)
        # value = Lambda(lambda v: K.tile(v, (1, self.action_size)))(v)
        #
        # policy = Add()([advt, value])

        #value and advt
        fc0 = Dense(512, activation='relu')(f1)
        fc1 = Dense(256, activation='relu')(fc0)
        fc2 = Dense(self.action_size + 1, activation='linear')(fc1)

        policy = Lambda(lambda a: K.expand_dims(a[:, 0], axis=-1) + a[:, 1:] - K.mean((a[:, 1:]), axis=1, keepdims=True),\
                        output_shape=(self.action_size, ))(fc2)

        model = Model(input=input, output=policy)
        model.compile(loss='mse',
                      optimizer=RMSprop(lr=self.learning_rate))

        # plot_model(model, to_file="Learning/Weights/ddqn.png", show_shapes=True)

        return model


class DQNPrioReplay(deep_q_net):

    def __init__(self,
                 state_size,
                 action_size,
                 path="Learning/Weights/weights_PRM.h5",
                 new_weights=True,
                 memory_size=100000,
                 replay_start_size=6000,
                 epsilon=1,
                 epsilon_min=.05,
                 max_step_for_epsilon_decay=250000):

        super(DQNPrioReplay, self).__init__(state_size,
                                            action_size,
                                            path,
                                            new_weights,
                                            memory_size,
                                            replay_start_size,
                                            epsilon,
                                            epsilon_min,
                                            max_step_for_epsilon_decay)

        self.memory = prm.Memory(memory_size)

    def remember(self, state, action, reward, next_state, done):
        exp = state, action, reward, next_state, done
        self.memory.store(exp)

    def get_minibatch(self, batch_size):
        idx, minibatch, is_weights = self.memory.sample(batch_size)
        return idx, np.reshape(minibatch, newshape=(batch_size, 5)), is_weights

    def train(self, states, target, is_weights, tree_idx, actions):
        is_weights = np.reshape(is_weights, newshape=(is_weights.shape[0]))
        self.model.fit(states, target, sample_weight=is_weights, verbose=1)

        abs_errs = np.abs(self.model.predict_on_batch(states)[np.arange(32), actions] - target[np.arange(32), actions])


        self.memory.batch_update(tree_idx, abs_errs)


class DuelDDQN(DoubleDQN):

    def _build_model(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.2

        K.tensorflow_backend.set_session(tf.Session(config=config))

        # Neural Net for Deep-Q learning Model

        input = Input(shape=self.state_size)

        h1 = Conv2D(16, 5, strides=1,
                    activation='relu',
                    kernel_initializer='he_normal',
                    #data_format="channels_first",
                    padding='same')(input)
        h2 = Conv2D(32, 3, strides=1,
                    activation='relu',
                    kernel_initializer='he_normal',
                    padding='same')(h1)

        f1 = Flatten()(h2)

        #value and advt
        fc0 = Dense(64, activation='relu')(f1)
        #fc1 = Dense(256, activation='relu')(fc0)
        fc2 = Dense(self.action_size + 1, activation='linear')(fc0)

        policy = Lambda(lambda a: K.expand_dims(a[:,0], axis=-1) + a[:, 1:] - K.mean((a[:, 1:]), axis=1, keepdims=True),\
                        output_shape=(self.action_size, ))(fc2)

        model = Model(input=input, output=policy)
        model.compile(loss='mse',
                      optimizer=RMSprop(lr=self.learning_rate))

        # plot_model(model, to_file="Learning/Weights/ddqn.png", show_shapes=True)

        return model


class DQNFullyConv(deep_q_net):

    def _build_model(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.5

        K.tensorflow_backend.set_session(tf.Session(config=config))

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
                         padding='same'))
        # model.add(Convolution2D(64, 3, strides=1, activation='relu'))
        #model.add(Flatten())
        #model.add(Dense(512, activation='relu', kernel_initializer='he_normal'))
        #model.add(Dense(self.action_size, activation='linear', kernel_initializer='he_normal'))

        model.add(Conv2D(1, 1, 1))


        model.compile(loss=self._huber_loss,  # 'mse',
                      # optimizer=Adam(lr=self.learning_rate))
                      optimizer=RMSprop(lr=self.learning_rate))

        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.state_size[0]), random.randrange(self.state_size[0]), 0

        act_values = self.model.predict(state)[0]

        x, y, c = np.unravel_index(np.argmax(act_values), act_values.shape)
        # index_1d = np.argmax(act_values)

        return x, y, c

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

        y = np.ndarray(shape=(32, self.state_size[0], self.state_size[1], 1))
        actions = np.ndarray(shape=(32, 3), dtype=int)
        i = 0
        for state, action, reward, next_state, done in minibatch:
            # self.model.fit(state, target_f, epochs=1, verbose=0)

            state_array[i] = state
            y[i] = self.set_target(reward, state, next_state, action, done)
            actions[i] = action #int(action)
            i += 1
            # self.model.fit(state, target_f, epochs=1, verbose=0)

        # self.model.fit(prediction, y, batch_size=1)
        # self.model.train_on_batch(state_array, y)
        self.train(state_array, y, is_weights, tree_idx, actions)

        self.decrease_epsilon_linear()
