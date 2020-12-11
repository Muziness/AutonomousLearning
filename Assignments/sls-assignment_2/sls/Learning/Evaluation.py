from collections import deque
from keras.callbacks import TensorBoard

import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
plt.ion()


class Evaluation:

    def __init__(self, avg_size=50, start_at=50):

        self.avg_size = avg_size
        self.memory = deque(maxlen=avg_size)
        self.epsilon_collection = deque()
        self.avg_collection = deque()
        self.start_at = start_at

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.line1, = self.ax.plot([], [], 'r')
        self.line2, = self.ax.plot([], [], 'b')
        plt.draw()

        print("Graph initialized")

        #print avg reward
        self.avg_rewards = deque()

    def moving_avg(self, reward, epsilon):
        self.memory.append(reward)

        if len(self.memory) >= self.start_at:
        # if True:

            avg = np.mean(self.memory)

            if len(self.memory) > 1 and self.memory[-1] * 10 < self.memory[-2]:
                print("Drop - overshoot?")

            self.avg_collection.append(avg)
            self.epsilon_collection.append(epsilon)

            x_axis = np.arange(self.start_at, len(self.avg_collection) + self.start_at)

            # self.line1.set_xdata(x_axis)
            # self.line1.set_ydata(self.avg_collection)

            self.line1.set_data(x_axis, self.avg_collection)

            # self.line2.set_xdata(x_axis)
            # self.line2.set_ydata(self.epsilon_collection)

            self.line2.set_data(x_axis, self.epsilon_collection)

            self.ax.relim()
            self.ax.autoscale_view(True, True, True)
            plt.draw()
            plt.pause(0.001)

            return

            plt.plot(x_axis, self.avg_collection, 'r', x_axis, self.epsilon_collection, 'b')
            plt.ylabel("AVG reward")
            plt.xlabel("Episodes")

            plt.show()

    def _get_avg(self):
        if len(self.avg_collection) > 0:
            return self.avg_collection[-1]
        else:
            return 0

    def save(self, path='Learning/Weights/fig.png'):
        self.fig.savefig(path)

    def print_avg_reward(self, reward_list):

        avg = np.mean(reward_list)
        self.avg_rewards.append(avg)

        x_axis = np.arange(0, len(self.avg_rewards))

        self.line1.set_xdata(x_axis)
        self.line1.set_ydata(self.avg_rewards)

        self.ax.relim()
        self.ax.autoscale_view(True, True, True)
        plt.draw()
        plt.pause(0.001)

    @staticmethod
    def create_tensorboard(log_dir="Learning/Summaries"):

        callback = TensorBoard(log_dir=log_dir,
                               histogram_freq=0,
                               write_grads=True,
                               write_images=True)

        return callback
