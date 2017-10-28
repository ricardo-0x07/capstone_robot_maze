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
        self.dir_sensors = {'u': ['l', 'u', 'r'], 'r': ['u', 'r', 'd'],
                    'd': ['r', 'd', 'l'], 'l': ['d', 'l', 'u'],
                    'up': ['l', 'u', 'r'], 'right': ['u', 'r', 'd'],
                    'down': ['r', 'd', 'l'], 'left': ['d', 'l', 'u']}
        self.dir_move = {'u': [0, 1], 'r': [1, 0], 'd': [0, -1], 'l': [-1, 0],
                    'up': [0, 1], 'right': [1, 0], 'down': [0, -1], 'left': [-1, 0]}
        self.dir_reverse = {'u': 'd', 'r': 'l', 'd': 'u', 'l': 'r',
                    'up': 'd', 'right': 'l', 'down': 'u', 'left': 'r'}
        self.heading = 'up'
        self.maze_dim = maze_dim
        self.next_waypoint = None
        self.valid_actions = ['left', 'forward', 'right']
        self.destination = None
        self.learning = True
        self.Q = dict()
        self.epsilon = 1.0
        self.alpha =0.5
        self.max_Q = 0.0
        self.rotation = 0
        self.movement = 0
        self.robot_pos = {'location': [0, 0], 'heading': 'up'}
        self.to_rotate = {'forward': 0, 'right': 90, 'left': -90}
        self.hit_goal = False
        self.goal_bounds = [self.maze_dim/2 - 1, self.maze_dim/2]
        self.outer_bounds = [(self.maze_dim/2 -2, y) for y in range(self.maze_dim/2 -2, self.maze_dim/2 +2)]
        self.outer_bounds.extend([(self.maze_dim/2 +1, y) for y in range(self.maze_dim/2 -2, self.maze_dim/2 +2)])
        self.outer_bounds.extend([(x, self.maze_dim/2 -2) for x in range(self.maze_dim/2 -2, self.maze_dim/2 +2)])
        self.outer_bounds.extend([(x, self.maze_dim/2 +1) for x in range(self.maze_dim/2 -2, self.maze_dim/2 +2)])
        self.outer_bounds = list(set(self.outer_bounds))
        # print 'self.outer_bounds: ', self.outer_bounds

        self.greater_bounds = [(self.maze_dim/2 -3, y) for y in range(self.maze_dim/2 -3, self.maze_dim/2 +3)]
        self.greater_bounds.extend([(self.maze_dim/2 +2, y) for y in range(self.maze_dim/2 -3, self.maze_dim/2 +3)])
        self.greater_bounds.extend([(x, self.maze_dim/2 -3) for x in range(self.maze_dim/2 -3, self.maze_dim/2 +3)])
        self.greater_bounds.extend([(x, self.maze_dim/2 +2) for x in range(self.maze_dim/2 -3, self.maze_dim/2 +3)])
        self.greater_bounds = list(set(self.greater_bounds))
        # print 'self.greater_bounds: ', self.greater_bounds

        self.even_greater_bounds = [(self.maze_dim/2 -4, y) for y in range(self.maze_dim/2 -4, self.maze_dim/2 +4)]
        self.even_greater_bounds.extend([(self.maze_dim/2 +3, y) for y in range(self.maze_dim/2 -4, self.maze_dim/2 +4)])
        self.even_greater_bounds.extend([(x, self.maze_dim/2 -4) for x in range(self.maze_dim/2 -4, self.maze_dim/2 +4)])
        self.even_greater_bounds.extend([(x, self.maze_dim/2 +3) for x in range(self.maze_dim/2 -4, self.maze_dim/2 +4)])
        self.even_greater_bounds = list(set(self.even_greater_bounds))
        # print 'self.even_greater_bounds: ', self.even_greater_bounds
        self.testing = False
        self.previous_action = (0, 0)
        self.state = (0, 1, 0, self.robot_pos['heading'], self.robot_pos['location'][0], self.robot_pos['location'][1])

    def re_init(self):
        '''
        Use the re-initialization function to set up attributes that your robot
        will use to navigate the maze. .
        '''
        self.heading = 'up'
        self.epsilon = 0.0
        self.alpha = 0.0
        self.next_waypoint = None
        self.valid_actions = ['left', 'forward', 'right']
        self.destination = None
        self.learning = True
        self.max_Q = 0.0
        self.rotation = 0
        self.movement = 0
        self.robot_pos = {'location': [0, 0], 'heading': 'up'}
        self.to_rotate = {'forward': 0, 'right': 90, 'left': -90}
        self.hit_goal = False
        self.testing = True
        self.previous_action = (0, 0)
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
        the maze) then returing the tuple ('Reset', 'Reset') will indicate to
        the tester to end the run and return the robot to the start.
        '''

        sense_list = []
        for value in sensors:
            if value >= 3:
                sense_list.append(4)
            else:
                sense_list.append(value + 1)
        self.valid_actions = []
        if sensors[0] > 0: 
            self.valid_actions.extend([(-90, move) for move in range(1, sense_list[0])])
        if sensors[1] > 0: 
            self.valid_actions.extend([(0, move) for move in range(1, sense_list[1])])
        if sensors[2] > 0: 
            self.valid_actions.extend([(90, move) for move in range(1, sense_list[2])])
        if sensors[0] == 0 and sensors[1] == 0 and sensors[2] == 0:
            self.valid_actions.extend([(rotate, 0) for rotate in [-90, 90]])


        self.state = (sensors[0], sensors[1], sensors[2], self.robot_pos['heading'], self.robot_pos['location'][0], self.robot_pos['location'][1])
        self.createQ(self.state)                 # Create 'state' in Q-table
        self.previous_action = self.choose_action(self.state)  # Choose an action
        if self.hit_goal:
            self.adjust()

        # Determine updated rotation
        self.rotation = self.previous_action[0]

        # Determine updated movement
        self.movement = self.previous_action[1]
        if self.epsilon <= 0 and not self.testing and self.hit_goal:
            print 'RESET'
            self.re_init()
            return 'Reset', 'Reset'

        reward = self.act() # Receive a reward
        self.learn(self.state, self.previous_action, reward)   # Q-learn
        if self.hit_goal:
            print 'reward', reward

        return self.rotation, self.movement

    def act(self):
        """
        Reward the agent for reaching goal and penalize them every other time.
        """
        # Determine current heading
        # perform rotation
        if self.rotation == -90:
            self.robot_pos['heading'] = self.dir_sensors[self.robot_pos['heading']][0]
        elif self.rotation == 90:
            self.robot_pos['heading'] = self.dir_sensors[self.robot_pos['heading']][2]
        elif self.rotation == 0:
            pass
        else:
            print "Invalid rotation value, no rotation performed."
        movement = self.movement
        while movement != 0:
            if self.movement > 0:
                self.robot_pos['location'][0] += self.dir_move[self.robot_pos['heading']][0]
                self.robot_pos['location'][1] += self.dir_move[self.robot_pos['heading']][1]
                movement -= 1
        for item in self.outer_bounds:
            if self.robot_pos['location'][0] == item[0] and self.robot_pos['location'][1] == item[1]:
                if self.movement < 1:
                    return -10
                else:
                    return 50

        for item in self.greater_bounds:
            if self.robot_pos['location'][0] == item[0] and self.robot_pos['location'][1] == item[1]:
                if self.movement < 1:
                    return -10
                else:
                    return 5

        for item in self.even_greater_bounds:
            if self.robot_pos['location'][0] == item[0] and self.robot_pos['location'][1] == item[1]:
                if self.movement < 1:
                    return -10
                else:
                    return .5

        if self.robot_pos['location'][0] in self.goal_bounds and self.robot_pos['location'][1] in self.goal_bounds:
            self.hit_goal = True
            return 1000
        else:
            if self.movement > 0:
                return -1
            else:
                return -10

    def adjust(self, destination=None, testing=False):
        """ The reset function is called at the beginning of each trial.
            'testing' is set to True if testing trials are being used
            once training trials have completed. """

        # Update epsilon using a decay function of your choice
        if self.epsilon > 0.0:
            self.epsilon = self.epsilon - 0.1
        # Update additional class parameters as needed
        # If 'testing' is True, set epsilon and alpha to 0
        if self.testing == True:
            self.epsilon = 0.0
            self.alpha = 0.0
        return True

    def build_state(self, sensors):
        state = sensors
        return state

    def get_maxQ(self, state):
        if state in self.Q:
            stateQValues = [self.Q[state][action] for action in self.valid_actions if action in self.Q[state]]
            a = np.array(stateQValues)
            if len(a) > 0:
                self.max_Q = a[np.argmax(a)]
        return self.max_Q

    def createQ(self, state):
        if self.learning == True:
            if state not in self.Q:
                self.Q[state] = {}
                for action in self.valid_actions:
                    self.Q[state][action] = 0.0
        return

    def choose_action(self, state, action=None):
        self.state = state
        if self.learning:
            number = random.random()
            if number <= self.epsilon:
                # print 'random_actions'
                action = random.choice(self.valid_actions)
            else:
                # print 'possible_actions'
                possible_actions = [action for action in self.valid_actions if action in self.Q[state] and self.Q[state][action] >= self.get_maxQ(state)]
                if len(possible_actions) > 0:
                    action = random.choice(possible_actions)
                else:
                    action = random.choice(self.valid_actions)
        else:
            action = random.choice(self.valid_actions)
        return action

    def learn(self, state, action, reward):
        maxQ = self.get_maxQ(state)
        if self.learning == True and state in self.Q and action in self.Q[state] and self.alpha > 0:
            updated_reward = reward * self.alpha + maxQ * (1 - self.alpha)
            self.Q[state][action] = updated_reward
