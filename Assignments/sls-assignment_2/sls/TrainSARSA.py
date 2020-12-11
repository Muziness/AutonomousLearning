# from pysc2.agents import base_agent
from pysc2.env import sc2_env, run_loop
from pysc2.lib import actions, features, units
from absl import app
from scipy.spatial import distance

from MiniGames.MyBaseAgent import MyBaseAgent
import MiniGames.Actions as Act
import Learning.Q_Learning as ql
import Learning.SARSA as sarsa
from MiniGames.utils import get_marine, get_beacon, state_of_marine

import math

_SCREEN = 32
_MINIMAP = 32
_EPISODES = 1000
_LEARN = True
_SAVE_Q_EVERY_STEP = False
_VISUALIZE = False


class MoveToBeaconAgent(MyBaseAgent):

    # To discretise the distance from Marine to Beacon
    _DISTANCE_WINDOW = 5

    def __init__(self, noq=False):
        super(MoveToBeaconAgent, self).__init__()

        if _LEARN:
            epsilon = 0.5
        else:
            epsilon = 0

        self.old_reward = 0

        if noq:
            return

        self.ql = ql.Q_Learning([action for action in Act.Actions],
                                epsilon=epsilon,
                                new_table=_LEARN,
                                save_table_ever_step=_SAVE_Q_EVERY_STEP,
                                descending_epsilon=True,
                                descend_epsilon_until=_EPISODES,
                                path="Learning/Q_Tables/q_table.pkl")

    def step(self, obs):
        super(MoveToBeaconAgent, self).step(obs)

        if obs.first():
            self.ql.descend_epsilon()
            return actions.FUNCTIONS.select_army("select")

        marine = get_marine(obs)
        beacon = get_beacon(obs)

        state = state_of_marine(marine, beacon, _SCREEN, self._DISTANCE_WINDOW)

        if _LEARN:
            chosen_action = self.ql.get_action(state, self.calc_reward())
        else:
            chosen_action = self.ql.choose_action(state)

        # if chosen_action.value[0] not in obs.observation.available_actions:
        #     self.old_reward = -1
        #     return actions.FUNCTIONS.no_op()

        return Act.action_to_function(chosen_action, marine, _SCREEN)


class MoveToBeaconSarsa(MoveToBeaconAgent):

    def __init__(self):
        super(MoveToBeaconSarsa, self).__init__(noq=True)

        self.ql = sarsa.SARSA([action for action in Act.Actions],
                              epsilon=0.5,
                              new_table=_LEARN,
                              save_table_ever_step=_SAVE_Q_EVERY_STEP,
                              descending_epsilon=True,
                              descend_epsilon_until=_EPISODES)


def main(unused_argv):

    # agent = MoveToBeaconAgent()
    agent = MoveToBeaconSarsa()

    try:
            """
            aif = features.AgentInterfaceFormat(
                feature_dimensions=features.Dimensions(screen=_SCREEN, minimap=_MINIMAP),
                # rgb_dimensions=features.Dimensions(screen=512, minimap=128),
                use_feature_units=True,
                action_space=actions.ActionSpace.FEATURES
            )
            """
            with sc2_env.SC2Env(
                map_name="MoveToBeacon",
                players=[sc2_env.Agent(sc2_env.Race.terran)],
                agent_interface_format=features.AgentInterfaceFormat(
                    feature_dimensions=features.Dimensions(screen=_SCREEN,
                                                           minimap=_MINIMAP),
                    use_feature_units=True
                ),
                step_mul=8,
                game_steps_per_episode=0,
                visualize=_VISUALIZE
            ) as env:

                run_loop.run_loop([agent], env, max_episodes=_EPISODES)

                if _LEARN:
                    agent.ql.save_q_table()

                """
                while True:

                    agent.setup(env.observation_spec(), env.action_spec())

                    timesteps = env.reset()
                    agent.reset()

                    while True:
                        step_actions = [agent.step(timesteps[0])]
                        if timesteps[0].last():
                            break
                        timesteps = env.step(step_actions)

                    agent.ql.save_q_table()
                    # agent.ql.print_q()

                """

    except KeyboardInterrupt:
        print("Exception")
        agent.ql.save_q_table()
        pass


if __name__ == "__main__":
    app.run(main)
