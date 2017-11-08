import random
import math
import numpy as np
import sys


class Robot(object):
    def __init__(self, maze_dim, astar=True, path_weight=1, estimate_weight=22):
        '''
        Use the initialization function to set up attributes that your robot
        will use to learn and navigate the maze. Some initial attributes are
        provided based on common information, including the size of the maze
        the robot is placed in.
        '''
        # List of action to support taking th agent back to the start
        self.reverse_actions = []
        self.isReversing = False
        self.astar = astar
        self.path_weight = path_weight
        self.estimate_weight = estimate_weight
        # Dictionaries to aid navigation
        self.dir_sensors = {'u': ['l', 'u', 'r'], 'r': ['u', 'r', 'd'],
                    'd': ['r', 'd', 'l'], 'l': ['d', 'l', 'u'],
                    'up': ['l', 'u', 'r'], 'right': ['u', 'r', 'd'],
                    'down': ['r', 'd', 'l'], 'left': ['d', 'l', 'u']}
        self.dir_rotation = {'u': [-90, 0, 90], 'r': [-90, 0, 90],
                    'd': [-90, 0, 90], 'l': [-90, 0, 90],
                    'up': [-90, 0, 90], 'right': [-90, 0, 90],
                    'down': [-90, 0, 90], 'left': [-90, 0, 90]}
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
        self.state = (tuple(self.robot_pos['location']))
        self.explored = []
        initial_path = (self.getStartStateLocation(),)
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
        self.state = (tuple(self.robot_pos['location']))
        # self.state = (tuple(self.robot_pos['location']), self.robot_pos['heading'], self.action)

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
        if self.isGoalState(self.robot_pos['location']):
            self.path = list(self.path[1:])
            self.re_init()
            return 'Reset', 'Reset'

        if self.testing:
            action = self.navigate()
            self.act(action)
            return action[0], action[1]

        if not self.isReversing:
            if self.astar:
                self.state, self.path = self.perform_astar(sensors)
            else:
                self.state, self.path = self.perform_bfs(sensors)
            self.new_path_previous_state = self.path[len(self.path)-2]
            self.switch_paths(self.previous_path, self.path)

        if self.robot_pos['location'][0] == self.new_path_previous_state[0] and self.robot_pos['location'][1] == self.new_path_previous_state[1] :
            self.isReversing = False
            action = self.compute_next_action(self.state)
            rotation = action[0]
            movement = action[1]
            self.previous_path = list(self.path)
            self.act(action)
            return rotation, movement
        else:
            self.isReversing = True
            action = self.reverse()
            self.act(action)
            rotation = action[0]
            movement = action[1]
            return rotation, movement

    def compute_action(self, successor, predecessor=None):
        index = self.dir_sensors[predecessor[1]].index(successor[1][:1])
        rotation = self.dir_rotation[predecessor[1]][index]
        movement = 0
        if successor[1] == 'u' or successor[1] == 'up':
            move = successor[0][1] - predecessor[0][1]

        if successor[1] == 'd' or successor[1] == 'down':
            move = predecessor[0][1] - successor[0][1]

        if successor[1] == 'r' or successor[1] == 'right':
            move = successor[0][0] - predecessor[0][0]

        if successor[1] == 'l' or successor[1] == 'left':
            move = predecessor[0][0] - successor[0][0]
        return rotation, move

    def compute_next_action(self, successor, predecessor=None):
        if predecessor == None:
            predecessor = self.robot_pos
        move = 0
        if predecessor['location'][0] > successor[0]:
            new_heading = 'l'
        elif predecessor['location'][0] < successor[0]:
            new_heading = 'r'
        elif predecessor['location'][1] > successor[1]:
            new_heading = 'd'
        elif predecessor['location'][1] < successor[1]:
            new_heading = 'u'
        else:
            new_heading = predecessor['heading']

        if new_heading == 'u':
            move = successor[1] - predecessor['location'][1]

        if new_heading == 'd':
            move = predecessor['location'][1] - successor[1]

        if new_heading == 'r':
            move = successor[0] - predecessor['location'][0]

        if new_heading == 'l':
            move = predecessor['location'][0] - successor[0]

        if new_heading in self.dir_sensors[predecessor['heading']]:
            index = self.dir_sensors[predecessor['heading']].index(new_heading)
            rotation = self.dir_rotation[predecessor['heading']][index]
        else:
            rotation = 0
            move = move * -1
        return rotation, move

    def compute_next_sim_action(self, successor, predecessor=None, heading='u'):
        if predecessor == None:
            predecessor = self.robot_pos
        move = 0
        if predecessor[0] > successor[0]:
            new_heading = 'l'
            move = predecessor[0] - successor[0]
        elif predecessor[0] < successor[0]:
            new_heading = 'r'
            move = successor[0] - predecessor[0]
        elif predecessor[1] > successor[1]:
            new_heading = 'd'
            move = predecessor[1] - successor[1]
        elif predecessor[1] < successor[1]:
            new_heading = 'u'
            move = successor[1] - predecessor[1]
        else:
            new_heading = heading

        if new_heading in self.dir_sensors[heading]:
            index = self.dir_sensors[heading].index(new_heading)
            rotation = self.dir_rotation[heading][index]
        else:
            rotation = 0
            move = move * -1
        return rotation, move, heading

    def getIndexOfState(self, previous_path, current_path):
        for less in range(1, len(previous_path)+1):
            state = previous_path[len(previous_path) - less]
            for index, item in enumerate(current_path):
                if item[0] == state[0] and item[1] == state[1]:
                    return index
        return -1

    def switch_paths(self, previous_path, current_path):
        """
        """
        index = self.getIndexOfState(previous_path, current_path)
        path_section = list(current_path[index:len(current_path)-1])
        self.reverse_actions = []
        self.switch_path = []
        while len(path_section) > 0:
            self.switch_path.append(path_section.pop())
        for state in previous_path[index+1:]:
            self.switch_path.append(state)

        self.switch_path = self.switch_path[:len(self.switch_path)-1]

    def reverse(self):
        """
        Reverse the agent to the start.
        return: action for the next reverse move
        """
        state = self.switch_path.pop()
        action = self.compute_next_action(state)
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

    def navigate(self):
        """
        Navigate the agent to the goal.
        return: action for the next move
        """
        successor = self.path.pop(0)
        action = self.compute_next_action(successor)
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
        self.successors = self.getSuccessors(self.robot_pos, sensors)
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

    def getSuccessors(self, robot_pos, sensors):
        """
        Get state successors
        robot_pos: robot position dictionary for which successors are to be found
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
            location = [robot_pos['location'][0], robot_pos['location'][1]]
            heading = self.dir_sensors[robot_pos['heading']][0]
            movement = max(min(int(sensors[0]), 1), 0) # fix to range [-3, 3]
            while movement:
                location[0] += self.dir_move[heading][0]
                location[1] += self.dir_move[heading][1]
                movement -= 1
            successors.append((tuple(location)))
            # successors.append((tuple(location),heading))
        if sensors[1] > 0: 
            location = [robot_pos['location'][0], robot_pos['location'][1]]
            heading = robot_pos['heading']
            movement = max(min(int(sensors[1]), 1), 0) # fix to range [-3, 3]
            while movement:
                location[0] += self.dir_move[heading][0]
                location[1] += self.dir_move[heading][1]
                movement -= 1
            successors.append((tuple(location)))
            # successors.append((tuple(location),heading))
        if sensors[2] > 0: 
            location = [robot_pos['location'][0], robot_pos['location'][1]]
            heading = self.dir_sensors[robot_pos['heading']][2]
            movement = max(min(int(sensors[2]), 1), 0) # fix to range [-3, 3]
            while movement:
                location[0] += self.dir_move[heading][0]
                location[1] += self.dir_move[heading][1]
                movement -= 1
            successors.append((tuple(location)))
            # successors.append((tuple(location),heading))
        return   successors      

    #### A STAR  SEARCH ###
    def perform_astar(self, sensors):
        """
        Search the node that has the lowest combined cost and heuristic first.
        path: list of states on a path.
        state: explored state/current state.
        return: path: lists of states from start to current state,  State: explored state/current state. 
        """
        self.update(sensors, self.path, self.state)
        return self.get_next_astar_state()
        
    def get_manhattan_cost_to_goal(self,location):
        """
        Compute the manhattan cost from a location to the goal.
        location: tuple of the coordinates for the location in the maze.
        return: the computed cost value.
        """
        total = 0
        cost = 0
        # D = 22
        for cell in self.goal_cells:
            dx = abs(cell[0]- location[0])
            dy = abs(cell[1]- location[1])
            total += (dx + dy)
        cost = self.estimate_weight * (total/len(self.goal_cells))
        return cost        

    def get_path_cost(self, path):
        cost = 0
        length = len(path)
        window_size = 2
        nb_pairs = length - window_size
        heading = 'u'
        for pair_marker in range(nb_pairs):
            pair = path[pair_marker:pair_marker + window_size]
            rotation, move, heading = self.compute_next_sim_action(pair[1], pair[0], heading=heading)
            if rotation == 0:
                cost += 1
            else:
                cost += self.path_weight
        return cost

    def get_next_astar_state(self):
        """
        Search the node that has the lowest combined cost and heuristic first.
        path: list of states on a path.
        state: explored state/current state.
        return: path: lists of states from start to current state,  State: explored state/current state. 
        """
        path = tuple()
        state = (0,0)

        previous_a_star = 999999
        sorted_frontier = sorted(self.frontier,key=len, reverse=False)

        for next_path in self.frontier:
            current_action_cost = self.get_manhattan_cost_to_goal(next_path[len(next_path)-1])
            current_a_star =  current_action_cost + self.get_path_cost(next_path)
            if current_a_star <= previous_a_star:
                previous_a_star = current_a_star
                path = next_path
        if path:
            self.frontier.remove(path)
            state = path[len(path) - 1]
        return state, path
    
    ### BREADTH FIRST SEARCH ###
    def perform_bfs(self, sensors):
        """
        Search the shallowest nodes in the search tree first.
        path: list of states on a path.
        state: explored state/current state.
        return: path: lists of states from start to current state,  State: explored state/current state. 
        """
        self.update(sensors, self.path, self.state)
        return self.get_next_bfs_state()
    def get_next_bfs_state(self):
        """
        Search the shallowest nodes in the search tree first.
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

