from Learning.Q_Learning import Q_Learning


class SARSA(Q_Learning):

    '''On policy learning'''

    def __init__(self,
                 actions,
                 epsilon=0.1,
                 gamma=0.9,
                 alpha=0.2,
                 new_table=True,
                 save_table_ever_step=False,
                 descending_epsilon=False,
                 epsilon_min=0.1,
                 descend_epsilon_until=0,
                 path="Learning/Q_Tables/sarsa_table.pkl"):

        super(SARSA, self).__init__(actions,
                                    epsilon,
                                    gamma,
                                    alpha,
                                    new_table,
                                    save_table_ever_step,
                                    descending_epsilon,
                                    epsilon_min,
                                    descend_epsilon_until,
                                    path)

    def get_action(self, state, reward_for_last_action):

        state_as_string = str(state)

        next_action = self.choose_action(state_as_string)

        if self.old_state_action_pair is not None:
            self.learn(self.old_state_action_pair, reward_for_last_action, state_as_string, next_action)

        if self.save_table_every_step:
            self.save_q_table()

        self.old_state_action_pair = (state_as_string, next_action)

        return next_action

    def learn(self, state_action_pair, reward, new_state, new_action):
        # next_actions = [self.get_q(new_state, a, 1) for a in self.actions]
        # best_next_action = max(next_actions)
        q_val_next_action = self.get_q(new_state, new_action, 1)
        q_value_next_state = self.gamma * q_val_next_action
        self.learn_q(state_action_pair, reward, q_value_next_state)
