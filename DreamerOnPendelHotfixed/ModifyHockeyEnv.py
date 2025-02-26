from hockey.hockey_env import HockeyEnv, CENTER_X, CENTER_Y

class ModifiedHockeyEnv(HockeyEnv):
    def set_state(self, state):
        """ function to revert the state of the environment to a previous state (observation)"""
        self.player1.position = (state[[0, 1]] + [CENTER_X, CENTER_Y]).tolist()
        self.player1.angle = state[2]
        self.player1.linearVelocity = [state[3], state[4]]
        self.player1.angularVelocity = state[5]
        self.player2.position = (state[[6, 7]] + [CENTER_X, CENTER_Y]).tolist()
        self.player2.angle = state[8]
        self.player2.linearVelocity = [state[9], state[10]]
        self.player2.angularVelocity = state[11]
        self.puck.position = (state[[12, 13]] + [CENTER_X, CENTER_Y]).tolist()
        self.puck.linearVelocity = [state[14], state[15]]
        if self.keep_mode:
          self.player1_has_puck = state[16]
          self.player2_has_puck = state[17]