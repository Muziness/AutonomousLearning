# from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features
from absl import app

from Learning.DQN import deep_q_net
from MiniGames.utils import move_screen, get_marine, preprocess_channels
from MiniGames.MyBaseAgent import MyBaseAgent
from Learning.Evaluation import Evaluation

import MiniGames.Actions as Act
import numpy as np
import math

_SCREEN = 32
_MINIMAP = 32
_ACTION_RANGE = 8

_VISUALIZE = True
_EPISODES = 2000


class MoveToBeaconAgent(MyBaseAgent):

    def __init__(self, state_size=_SCREEN):
        super(MoveToBeaconAgent, self).__init__()

        self.old_state = None
        self.old_action = None
        self.dqn = deep_q_net(state_size=(2, state_size, state_size),
                              action_size=(2 * _ACTION_RANGE + 1) * (2 * _ACTION_RANGE + 1),
                              new_weights=True,
                              epsilon=1)

    def step(self, obs):
        super(MoveToBeaconAgent, self).step(obs)

        if obs.first():
            return actions.FUNCTIONS.select_army("select")

        state = get_state(obs)

        if self.old_state is not None:
            self.dqn.remember(self.old_state, self.old_action, self.calc_reward(), state, False)
            self.dqn.replay(32)

        action = self.dqn.act(state)

        self.old_state = state
        self.old_action = action

        return self.transform_action_to_function(action, obs)

    @staticmethod
    def transform_action_to_function_whole_field(action):
        y = math.floor(action / _SCREEN)
        x = action % _SCREEN
        return move_screen(x, y)

    @staticmethod
    def transform_action_to_function(action, obs):
        y = math.floor(action / (2 * _ACTION_RANGE + 1))
        y -= (_ACTION_RANGE + 1)

        x = action % (2 * _ACTION_RANGE + 1)
        x -= (_ACTION_RANGE + 1)

        marine = get_marine(obs)

        new_x = marine.x + x
        new_x = MoveToBeaconAgent.check_bounds(new_x)

        new_y = marine.y + y
        new_y = MoveToBeaconAgent.check_bounds(new_y)

        return move_screen(new_x, new_y)

    @staticmethod
    def check_bounds(coord):
        if coord > _SCREEN - 1:
            coord = _SCREEN - 1
        elif coord < 0:
            coord = 0
        return coord



def transform_action_to_function(action, obs):

    screen = obs.observation.feature_screen.shape[1]

    marine = get_marine(obs)

    if action == 0:
        return Act.action_to_function(Act.Actions.down_left, marine, screen)
    elif action == 1:
        return Act.action_to_function(Act.Actions.down, marine, screen)
    elif action == 2:
        return Act.action_to_function(Act.Actions.down_right, marine, screen)
    elif action == 3:
        return Act.action_to_function(Act.Actions.up_left, marine, screen)
    elif action == 4:
        return Act.action_to_function(Act.Actions.up, marine, screen)
    elif action == 5:
        return Act.action_to_function(Act.Actions.up_right, marine, screen)
    elif action == 6:
        return Act.action_to_function(Act.Actions.right, marine, screen)
    elif action == 7:
        return Act.action_to_function(Act.Actions.left, marine, screen)


def get_state(obs, channels=2):
    state_size = obs.observation.feature_screen.shape
    state = np.ndarray(shape=(channels, state_size[1], state_size[2]))

    if channels == 17:
        state = obs.observation.feature_screen
        state = np.asanyarray(state)
        state = preprocess_channels(obs)
    elif channels == 2:
        state[0] = obs.observation.feature_screen.player_relative
        #state[1] = obs.observation.feature_screen.selected
        # state[1] = obs.observation.feature_minimap.player_relative
        state[1] = obs.observation.feature_screen.unit_density
    elif channels == 1:
        state[0] = obs.observation.feature_screen.unit_density #player_relative

    state = state.transpose(1, 2, 0)
    state = state[np.newaxis, :]
    return state


def main(unused_argv):

    agent = MoveToBeaconAgent()

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
