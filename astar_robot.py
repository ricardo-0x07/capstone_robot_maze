import random
import math
import numpy as np
import sys


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
        self.total_time = 0
        self.run_active = True
        # Max time to complete both runs
        self.max_time = 1000
        # Dimension of the maze
        self.maze_dim = maze_dim
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
        self.previous_action = (0, 0)
        self.count = 0

        self.action = (0, 0)
        self.robot_pos = {'location': [0, 0], 'heading': 'up'}
        self.state = (tuple(self.robot_pos['location']), self.robot_pos['heading'], self.action)
        self.explored = []
        initial_path = ((self.getStartStateLocation(), 'up', self.action,),)
        self.frontier = set()
        self.previous_path = [self.state]
        self.path = tuple(initial_path)

    def re_init(self):
        '''
        Use the re-initialization function to set up attributes that your robot
        will use to navigate the maze. .
        '''
        self.total_time = 0
        self.hit_goal = False
        self.testing = True

        self.action = (0, 1)
        self.robot_pos = {'location': [0, 0], 'heading': 'up'}
        self.state = (tuple(self.robot_pos['location']), self.robot_pos['heading'], self.action)

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
        self.count += 1
        # if self.count >30:
        #     sys.exit()
        if self.isGoalState(self.robot_pos['location']):
            self.actions = [action[2] for action in self.path]
            self.re_init()
            return 'Reset', 'Reset'

        if self.testing:
            action = self.navigate()
            return action[0], action[1]

        if not self.isReversing:
            self.update(sensors, self.path, self.state)
            self.state, self.path = self.get_next_state()
            self.new_path_previous_state = self.path[len(self.path)-2]
        
        if self.robot_pos['location'][0] == self.new_path_previous_state[0][0] and self.robot_pos['location'][1] == self.new_path_previous_state[0][1] and self.robot_pos['heading'] == self.new_path_previous_state[1]:
            self.isReversing = False
            action = self.state[2]
            rotation = action[0]
            movement = action[1]
            self.previous_path = list(self.path)
            self.act(action)
            for pstate in self.previous_path:
                action = pstate[2]
                reverse_action1 = None
                reverse_action2 = None
                action_list = []
                reverse_action1 = (0, action[1]*-1)
                reverse_action2 = (action[0]* -1, 0)
                action_list = [reverse_action2, reverse_action1]
                self.reverse_actions.extend(action_list)
            return rotation, movement
        else:
            self.isReversing = True
            action = self.reverse()
            rotation = action[0]
            movement = action[1]
            return rotation, movement
                
    def reverse(self):
        """
        Reverse the agent to the start.
        return: action for the next reverse move
        """
        # print 'self.reverse_actions', self.reverse_actions
        action = self.reverse_actions.pop(len(self.reverse_actions) -1)
        self.act(action)
        return action

    def act(self, action):
        """
        Update robot's location and heading based on intended rotation and movement. 
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
        while movement:
            if movement > 0:
                self.robot_pos['location'][0] += self.dir_move[self.robot_pos['heading']][0]
                self.robot_pos['location'][1] += self.dir_move[self.robot_pos['heading']][1]
                movement -= 1
            else:
                rev_heading = self.dir_reverse[self.robot_pos['heading']]
                self.robot_pos['location'][0] += self.dir_move[rev_heading][0]
                self.robot_pos['location'][1] += self.dir_move[rev_heading][1]
                movement += 1

    def get_next_state(self):
        """
        Retrieve the deepest path and it's frontier state from the frontier.
        path: list of states on a path.
        state: explored state/current state.
        return: path: lists of states from start to current state,  State: explored state/current state. 
        """
        path = tuple()
        sorted_frontier = sorted(self.frontier,key=len, reverse=False)
        path = sorted_frontier.pop()
        self.frontier.remove(path)

        if path:
            state = path[len(path) - 1]
        return state, path

    def update(self, sensors, path, state):
        """
        Update the list of explored and frontier states.
        sensors: a list of three distances from the robot's left, front, and
        right-facing sensors, in that order.
        path: list of states on a path.
        state: explored state/current state.
        return: path: lists of states from start to current state,  State: explored state/current state. 
        """
        self.explored.append(state)
        self.successors = self.getSuccessors(state, sensors)
        frontier_states = [frontier_path[len(frontier_path) - 1]
                            for frontier_path in self.frontier if frontier_path]
        for successor in self.successors:
            if successor not in self.explored:
                if successor not in frontier_states:
                    copy = list(path)
                    copy.append(successor)
                    copy = tuple(copy)
                    self.frontier.add(copy)
        return state, path

    def navigate(self):
        """
        Navigate the agent to the goal.
        return: action for the next move
        """
        action = self.actions.pop(0)
        return action

    def getStartStateLocation(self):
        """
        return: start state object.
        """
        return (0,0)

    def isGoalState(self, location):
        """
        Check if a location is in the goal state.
        location: location coordinate list.
        return: boolean True/False
        """
        if location[0] in self.goal_bounds and location[1] in self.goal_bounds:
            return True
        else:
            return False

    def getSuccessors(self, state, sensors):
        """
        Get state successors
        state: state object for which successors are to be found
        sensors: list of sensor readings, distance to next wall to left, forward or right.
        return: list of successors
        """
        sense_list = []
        for value in sensors:
            if value >= 3:
                sense_list.append(4)
            else:
                sense_list.append(value + 1)
        successors = []
        if sensors[0] > 0: 
            location = list(state[0])
            heading = self.dir_sensors[state[1]][0]
            location[0] += self.dir_move[heading][0]
            location[1] += self.dir_move[heading][1]
            successors.append((tuple(location),heading,(-90, 1)))
        if sensors[1] > 0: 
            location = list(state[0])
            heading = state[1]
            location[0] += self.dir_move[heading][0]
            location[1] += self.dir_move[heading][1]
            successors.append((tuple(location),heading,(0, 1)))
        if sensors[2] > 0: 
            location = list(state[0])
            heading = self.dir_sensors[state[1]][2]
            location[0] += self.dir_move[heading][0]
            location[1] += self.dir_move[heading][1]
            successors.append((tuple(location),heading,(90, 1)))
        return   successors      
