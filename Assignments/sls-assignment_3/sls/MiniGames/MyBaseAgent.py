from pysc2.agents import base_agent


class MyBaseAgent(base_agent.BaseAgent):

    def __init__(self):
        super(MyBaseAgent, self).__init__()

        self.old_reward = 0

    def calc_reward(self):

        if self.old_reward == -1:
            self.old_reward = self.reward
            return -1
        elif self.reward > self.old_reward:
            self.old_reward = self.reward
            return 1
        else:
            return 0
