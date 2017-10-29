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
        self.robot_pos = {'location': (0, 0), 'heading': 'up'}
        self.to_rotate = {'forward': 0, 'right': 90, 'left': -90}
        self.hit_goal = False
        self.goal_bounds = [self.maze_dim/2 - 1, self.maze_dim/2]

        self.testing = False
        self.previous_action = (0, 0)
        self.state = self.robot_pos['location']

        initial_path = ((self.getStartState(), 'up', 0,),)
        self.frontier = set([initial_path])
        self.explored = []

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
        the maze) then returning the tuple ('Reset', 'Reset') will indicate to
        the tester to end the run and return the robot to the start.
        '''
        # initial_path = ((self.getStartState(), 'up', 0,),)
        # self.frontier = set([initial_path])
        # self.explored = []
        path = tuple()
        # while True:
        # if not self.frontier:
        #     return []
        path = self.frontier.pop()
        if path:
            state = path[len(path) - 1]
            print 'state: ', state
        if self.isGoalState(state[0]):
            self.actions = [action[1] for action in path]

        self.explored.append(state)
        actions = self.getSuccessors(state, sensors)
        frontier_states = [frontier_path[len(frontier_path) - 1] for frontier_path in self.frontier if frontier_path]
        for action in actions:
            if action not in self.explored:
                if action not in frontier_states:
                    copy = list(path)
                    copy.append(action)
                    copy = tuple(copy)
                    self.frontier.add(copy)
        return self.rotation, self.movement

    def getStartState(self):
        return ((0,0), 'up', 0,),)

    def isGoalState(self, location):
        if location[0] in self.goal_bounds and location[1] in self.goal_bounds:
            self.hit_goal = True
            return self.hit_goal

    def getSuccessors(self, state, sensors):
        location = state[0]
        heading = state[1]
        successors = []
        for value in sensors:
            if value > 0:
                successors.append(((location[0]+= self.dir_move[heading][0], location[1]+= self.dir_move[heading][1]), self.dir_sensors[heading][0], 1))
        return successors

    def act(self):
        """
        Update location and heading based on intended rotation and movement. 
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
                    return -1000
                else:
                    return 100

        for item in self.greater_bounds:
            if self.robot_pos['location'][0] == item[0] and self.robot_pos['location'][1] == item[1]:
                if self.movement < 1:
                    return -1000
                else:
                    return 20

        for item in self.even_greater_bounds:
            if self.robot_pos['location'][0] == item[0] and self.robot_pos['location'][1] == item[1]:
                if self.movement < 1:
                    return -1000
                else:
                    return 15

        for item in self.next_even_greater_bounds:
            if self.robot_pos['location'][0] == item[0] and self.robot_pos['location'][1] == item[1]:
                if self.movement < 1:
                    return -1000
                else:
                    return 10

        for item in self.final_even_greater_bounds:
            if self.robot_pos['location'][0] == item[0] and self.robot_pos['location'][1] == item[1]:
                if self.movement < 1:
                    return -1000
                else:
                    return 5

        if self.robot_pos['location'][0] in self.goal_bounds and self.robot_pos['location'][1] in self.goal_bounds:
            # print "self.robot_pos['location']", self.robot_pos['location']
            # print 'self.goal_bounds', self.goal_bounds
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
        return (sensors[0], sensors[1], sensors[2], self.robot_pos['heading'], self.robot_pos['location'][0], self.robot_pos['location'][1])

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
