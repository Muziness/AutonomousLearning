# from pysc2.agents import base_agent
from pysc2.env import sc2_env, run_loop
from pysc2.lib import actions, features, units
from absl import app

from Learning.DQN import deep_q_net, DoubleDQN, DDQN, DQNPrioReplay, DuelDDQN, DQNFullyConv
from MiniGames.utils import move_screen, get_marine, preprocess_channels
from MiniGames.MoveToBeacon_DQN import MoveToBeacon8Actions
from Learning.Evaluation import Evaluation

import MiniGames.Actions as Act
import numpy as np


_SCREEN = 32
_MINIMAP = 32
_ACTION_RANGE = 8

_VISUALIZE = True
_EPISODES = 20000


def main(unused_argv):

    agent = MoveToBeacon8Actions(state_size=_SCREEN, dqn="dueling")

    evaluater = Evaluation(50)

    try:

            with sc2_env.SC2Env(
                map_name="MoveToBeacon",
                players=[sc2_env.Agent(sc2_env.Race.terran)],
                agent_interface_format=features.AgentInterfaceFormat(
                    feature_dimensions=features.Dimensions(screen=_SCREEN,
                                                           minimap=_MINIMAP),
                    use_feature_units=True
                ),
                step_mul=16,
                game_steps_per_episode=0,
                visualize=_VISUALIZE
            ) as env:

                # run_loop.run_loop([agent], env, max_episodes=1)

                episodes = 0

                while episodes <= _EPISODES:

                    episodes += 1

                    agent.setup(env.observation_spec(), env.action_spec())

                    timesteps = env.reset()
                    agent.reset()

                    while True:
                        step_actions = [agent.step(timesteps[0])]
                        if timesteps[0].last():
                            # agent.dqn.decrease_epsilon_factor()
                            # this is C
                            if episodes % 5 == 0:
                                agent.dqn.update_target()
                            # agent.dqn.replay(32)
                            evaluater.moving_avg(timesteps[0].observation.score_cumulative[0], agent.dqn.epsilon)
                            break
                        timesteps = env.step(step_actions)

                agent.dqn.save_weights()
                    # agent.ql.save_q_table()
                    # agent.ql.print_q()


    except KeyboardInterrupt:
        print("Exception")
        pass


if __name__ == "__main__":
    app.run(main)

# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 22:43:47 2020

@author: sreichhuber
"""


