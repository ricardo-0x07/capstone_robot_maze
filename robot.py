import random
import math
import numpy as np


class Robot(object):
    def __init__(self, maze_dim):
        '''
        Use the initialization function to set up attributes that your robot
        will use to learn and navigate the maze. Some initial attributes are
        provided based on common information, including the size of the maze
        the robot is placed in.
        '''
        # List of action to support taking th agent back to the start
        self.reverse_actions = []
        self.isReversing = False
        # Dictionaries to aid navigation
        self.dir_sensors = {'u': ['l', 'u', 'r'], 'r': ['u', 'r', 'd'],
                    'd': ['r', 'd', 'l'], 'l': ['d', 'l', 'u'],
                    'up': ['l', 'u', 'r'], 'right': ['u', 'r', 'd'],
                    'down': ['r', 'd', 'l'], 'left': ['d', 'l', 'u']}
        self.dir_move = {'u': [0, 1], 'r': [1, 0], 'd': [0, -1], 'l': [-1, 0],
                    'up': [0, 1], 'right': [1, 0], 'down': [0, -1], 'left': [-1, 0]}
        self.dir_reverse = {'u': 'd', 'r': 'l', 'd': 'u', 'l': 'r',
                    'up': 'd', 'right': 'l', 'down': 'u', 'left': 'r'}
        self.heading = 'up'
        self.total_time = 0
        self.run_active = True
        # Max time to complete both runs
        self.max_time = 1000
        # Dimension of the maze
        self.maze_dim = maze_dim
        # Valid action list
        self.valid_actions = []
        # Learning flag
        self.learning = True
        # Dictionary for the Q-table value function, where the optimal policy will be accumulated
        self.Q = dict()
        # Exploration factor used determine what propotion of the time to explore the maze or exploting whats has been learned 
        self.epsilon = 1.0
        # Learning rate, help determine how much to learn from the result of an action
        self.alpha =0.5
        # Max Q value 
        self.max_Q = 0.0
        self.robot_pos = {'location': [0, 0], 'heading': 'up'}
        # Initialize flag for if the agent has reached the goal
        self.hit_goal = False
        # Initialize goal bounds, used to check if agent reaches goal
        self.goal_bounds = [self.maze_dim/2 - 1, self.maze_dim/2]
        # Initialize list of cells in the goal state, used to compute rewards scaled by distance from goal state
        self.goal_cells = [(self.maze_dim/2 -1, y) for y in range(self.maze_dim/2 -1, self.maze_dim/2 +1)]
        self.goal_cells.extend([(self.maze_dim/2 +0, y) for y in range(self.maze_dim/2 -1, self.maze_dim/2 +1)])
        self.goal_cells.extend([(x, self.maze_dim/2 -1) for x in range(self.maze_dim/2 -1, self.maze_dim/2 +1)])
        self.goal_cells.extend([(x, self.maze_dim/2 +0) for x in range(self.maze_dim/2 -1, self.maze_dim/2 +1)])
        self.goal_cells = list(set(self.goal_cells))

        self.testing = False
        self.action = (0, 0)
        self.previous_action = (0, 0)
        self.state = (0, 1, 0, self.robot_pos['heading'], self.robot_pos['location'][0], self.robot_pos['location'][1])

    def re_init(self):
        '''
        Use the re-initialization function to set up attributes that your robot
        will use to navigate the maze. .
        '''
        self.total_time = 0
        self.previous_action = (0, 0)
        self.heading = 'up'
        self.epsilon = 0.0
        self.alpha = 0.0
        self.valid_actions = ['left', 'forward', 'right']
        self.learning = True
        self.max_Q = 0.0
        self.robot_pos = {'location': [0, 0], 'heading': 'up'}
        self.hit_goal = False
        self.testing = True
        self.action = (0, 0)
        self.state = (0, 1, 0, self.robot_pos['heading'], self.robot_pos['location'][0], self.robot_pos['location'][1])

    def reset(self):
        '''
        Use the reset parameters after the agent has reversed to the start to explore more possible paths
        '''
        self.previous_action = (0, 0)
        self.robot_pos = {'location': [0, 0], 'heading': 'up'}
        self.hit_goal = False
        self.state = (0, 1, 0, self.robot_pos['heading'], self.robot_pos['location'][0], self.robot_pos['location'][1])

    def next_move(self, sensors):
        '''
        Use this function to determine the next move the robot should make,
        based on the input from the sensors after its previous move. Sensor
        inputs are a list of three distances from the robot's left, front, and
        right-facing sensors, in that order.

        Outputs should be a tuple of two values. The first value indicates
        robot rotation (if any), as a number: 0 for no rotation, +90 for a
        90-degree rotation clockwise, and -90 for a 90-degree rotation
        counterclockwise. Other values will result in no rotation. The second
        value indicates robot movement, and the robot will attempt to move the
        number of indicated squares: a positive number indicates forwards
        movement, while a negative number indicates backwards movement. The
        robot may move a maximum of three units per turn. Any excess movement
        is ignored.

        If the robot wants to end a run (e.g. during the first training run in
        the maze) then returning the tuple ('Reset', 'Reset') will indicate to
        the tester to end the run and return the robot to the start.
        '''
        self.total_time += 1
        if self.hit_goal and not self.testing:
            self.adjust()
        if self.epsilon <= 0 and not self.testing:
            self.run_active = False
            self.isReversing = False
            self.re_init()
            return 'Reset', 'Reset'
        if self.isReversing:
            print 'self.isReversing'
            action = self.reverse()
        else:
            action = self.forward(sensors)
        self.previous_action = action

        return action[0], action[1]

    def forward(self, sensors):
        """
        Drives the agent forward towards the goal
        sensors: inputs are a list of three distances from the robot's left, front, and
        right-facing sensors, in that order.
        return: action for the next move
        """
        # Determine valid actions based on sensor readings
        self.valid_actions = self.get_valid_actions(sensors)

        # Build state
        self.state = self.build_state(sensors)
        self.createQ(self.state)                 # Create 'state' in Q-table
        action = self.choose_action(self.state)  # Choose an action
        reverse_action1 = None
        reverse_action2 = None
        action_list = []
        if action[0] != 0:
            action_list.append((action[0]* -1, 0))
        if action[1] != 0:
            action_list.append((0, action[1]*-1))
        reverse_action1 = (0, action[1]*-1)
        reverse_action2 = (action[0]* -1, 0)
        action_list = [reverse_action2, reverse_action1] 
        self.reverse_actions.extend(action_list)

        reward = self.act(self.previous_action) # Receive a reward
        # print 'reward', reward
        self.learn(self.state, action, reward)   # Q-learn
        return action

    def reverse(self):
        """
        Reverse the agent to the start.
        return: action for the next reverse move
        """
        self.hit_goal = False
        action = (0,0)
        if len(self.reverse_actions) == 0:
            self.isReversing = False
            self.reset()
            return action
        action = self.reverse_actions.pop(len(self.reverse_actions) -1)
        return action

    def act(self, action):
        """
        Update location and heading based on intended rotation and movement. 
        Reward the agent for reaching goal and penalize them every other time.
        """
        # Determine current heading
        # perform rotation
        rotation = action[0]
        if rotation == -90:
            self.robot_pos['heading'] = self.dir_sensors[self.robot_pos['heading']][0]
        elif rotation == 90:
            self.robot_pos['heading'] = self.dir_sensors[self.robot_pos['heading']][2]
        elif rotation == 0:
            pass
        else:
            print "Invalid rotation value, no rotation performed."
        movement = action[1]
        while movement != 0:
            if movement > 0:
                self.robot_pos['location'][0] += self.dir_move[self.robot_pos['heading']][0]
                self.robot_pos['location'][1] += self.dir_move[self.robot_pos['heading']][1]
                movement -= 1
        # print "self.robot_pos['location']", self.robot_pos['location']
        if self.robot_pos['location'][0] in self.goal_bounds and self.robot_pos['location'][1] in self.goal_bounds:
            self.hit_goal = True
            return 10000
        elif action[1] > 0:
            total = 0.0
            score = 0.0
            for cell in self.goal_cells:
                total += (cell[0]-self.robot_pos['location'][0])**2 + (cell[1]-self.robot_pos['location'][0])**2
            score = 100*(len(self.goal_cells)/total)
            # print 'len(self.goal_cells)', len(self.goal_cells)
            # print 'total', total
            # print 'score', score
            return score
        elif action[1] < 1:
            return -1000

    def check_reward(self, action):
        """
        Checks rewards for possible actions.
        action: a possible action
        return: reward value based on possible action
        """
        location = [self.robot_pos['location'][0], self.robot_pos['location'][1]]
        movement = action[1]
        while movement > 0:
            if movement > 0:
                location[0] += self.dir_move[self.robot_pos['heading']][0]
                location[1] += self.dir_move[self.robot_pos['heading']][1]
                movement -= 1
        if location[0] in self.goal_bounds and location[1] in self.goal_bounds:
            return 1000
        elif action[1] > 0:
            total = 0
            score = 0
            for cell in self.goal_cells:
                total += (cell[0]- location[0])**2 + (cell[1]- location[0])**2
            score = 1000/(total/len(self.goal_cells))
            return score
        elif action[1] < 1:
            return -1000

    def get_valid_actions(self, sensors):
        """
        Determines valid actions.
        sensors: tuple of sensor values.
        valid_actions: list of valid actions.
        """
        sense_list = []
        for value in sensors:
            if value >= 3:
                sense_list.append(4)
            else:
                sense_list.append(value + 1)
        valid_actions = []
        if sensors[0] > 0: 
            # valid_actions.append((-90, 1))
            value1 = 3
            if sensors[0] < 3:
                value1 = sensors[0]
            valid_actions.extend([(-90, move) for move in range(1,value1+1)])
        if sensors[1] > 0: 
            # valid_actions.append((0, 1))
            value2 = 3
            if sensors[1] < 3:
                value2 = sensors[1]
            valid_actions.extend([(0, move) for move in range(1,value2+1)])
        if sensors[2] > 0: 
            # valid_actions.append((90, 1))
            value3 = 3
            if sensors[2] < 3:
                value3 = sensors[2]
            valid_actions.extend([(90, move) for move in range(1,value3+1)])
        if sensors[0] == 0 and sensors[1] == 0 and sensors[2] == 0:
            valid_actions.extend([(rotate, 0) for rotate in [-90, 90]])
        return   valid_actions      

    def adjust(self, destination=None, testing=False):
        """ The reset function is called at the beginning of each trial.
            'testing' is set to True if testing trials are being used
            once training trials have completed. """

        # Update epsilon using a decay function of your choice
        if self.epsilon > 0.0:
            self.epsilon = self.epsilon - 0.5
        # Update additional class parameters as needed
        # If 'testing' is True, set epsilon and alpha to 0
        if self.testing == True:
            self.epsilon = 0.0
            self.alpha = 0.0
        return True

    def build_state(self, sensors):
        """
        Builds the state for the current locations and sensor readings.
        sensors: tuple of sensor readings for current location.
        return: built state.
        """
        return (sensors[0], sensors[1], sensors[2], self.robot_pos['heading'], self.robot_pos['location'][0], self.robot_pos['location'][1])

    def get_maxQ(self, state):
        """
        Determines the maximum q-value for the possible actions from a state.
        state: state object.
        returns: the max value for all of a state's possible actions.
        """
        if state in self.Q:
            stateQValues = [self.Q[state][action] for action in self.valid_actions if action in self.Q[state]]
            a = np.array(stateQValues)
            if len(a) > 0:
                self.max_Q = a[np.argmax(a)]
        return self.max_Q

    def createQ(self, state):
        """
        Initializes state dictionaries in the q-table.
        state: state object.abs.
        """
        if self.learning == True:
            if state not in self.Q:
                self.Q[state] = {}
                for action in self.valid_actions:
                    self.Q[state][action] = 0.0
        return

    def choose_action(self, state):
        """
        Chooses an action for the next move.
        state: current state object.
        return action of the the next move.
        """
        self.state = state
        if self.learning:
            number = random.random()
            if number <= self.epsilon:
                reward_values = [self.check_reward(action) for action in self.valid_actions]
                a = np.array(reward_values)
                if len(a) > 0:
                    max_reward = a[np.argmax(a)]
                possible_actions = [action for action in self.valid_actions if self.check_reward(action) >= max_reward]
                if len(possible_actions) > 0:
                    action = random.choice(possible_actions)
                    return action
                else:
                    action = random.choice(self.valid_actions)
                    return action
            else:
                possible_actions = [action for action in self.valid_actions if action in self.Q[state] and self.Q[state][action] >= self.get_maxQ(state)]
                if len(possible_actions) > 0:
                    action = random.choice(possible_actions)
                    return action
                else:
                    reward_values = [self.check_reward(action) for action in self.valid_actions]
                    a = np.array(reward_values)
                    if len(a) > 0:
                        max_reward = a[np.argmax(a)]
                    possible_actions = [action for action in self.valid_actions if self.check_reward(action) >= max_reward]
                    if len(possible_actions) > 0:
                        action = random.choice(possible_actions)
                        return action
                    else:
                        action = random.choice(self.valid_actions)
                        return action
        else:
            action = random.choice(self.valid_actions)
            return action

    def learn(self, state, action, reward):
        """
        Updates the Qtable dictionary.
        state: state object.
        action: choosen action.
        reward for choosen action.
        """
        maxQ = self.get_maxQ(state)
        if self.learning == True and state in self.Q and action in self.Q[state] and self.alpha > 0:
            updated_reward = reward * self.alpha + maxQ * (1 - self.alpha)
            self.Q[state][action] = updated_reward
        return
