# Machine Learning Engineer Nanodegree
## Capstone Project: Plot and Navigate a Virtual Maze

### Install

This project requires the following 
**Python 2.7** 
**Numpy** 

### Code

- robot.py - This script establishes the robot class.
- maze.py - This script contains functions for constructing the maze and for checking for walls upon robot movement or sensing.
- tester.py - This script will be run to test the robot’s ability to navigate mazes.
- showmaze.py - This script can be used to create a visual demonstration of what a maze looks like.
- test_maze_##.txt - Four  mazes upon which to test your robot. 

### Run

1. Adjust default params in the Robot Class init function def ```__init__(self, maze_dim, astar=True, path_weight=1, estimate_weight=22)``` as follows:
    - astar=True to use the A* model and astar=False to use the Breadth First model.
    - adjust the weight values as required to tune the A* Model.
2. In a terminal or command window, navigate to the top-level project directory and run one of the following commands to test robot on the mazes or to show the mazes:

```python tester.py test_maze_01.txt```  
```python showmaze.py test_maze_01.txt```

