# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from game import Directions
from util import PriorityQueue


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    # DFS generally uses stack to implement, FILO
    stack = util.Stack()
    # Stack stores items with two features: Current state and route from start state to current state
    stack.push([problem.getStartState(), []])
    visited_states = set()

    while True:
        current_state, path = stack.pop()
        # Check if current state is final goal, if true then return the result path
        if problem.isGoalState(current_state):
            return path

        if not current_state in visited_states:
            # Add current state to the set so that it won't be access twice
            visited_states.add(current_state)
            # Get successors information
            successors_tuple = problem.getSuccessors(current_state)
            # Add new state to stack
            for successor_state, action, cost in successors_tuple:
                if successor_state in visited_states:
                     continue
                next_path = path + [action]
                stack.push([successor_state, next_path])


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    # BFS generally uses stack to implement, FIFO
    queue = util.Queue()
    # Stack stores items with two features: Current state and route from start state to current state
    queue.push([problem.getStartState(), []])
    visited_states = set()

    while True:
        current_state, path = queue.pop()
        if problem.isGoalState(current_state):
            return path
        if not current_state in visited_states:
            visited_states.add(current_state)
            # Get successors information
            successors_tuple = problem.getSuccessors(current_state)
            # Add new state to stack
            for successor_state, action, cost in successors_tuple:
                if successor_state in visited_states:
                     continue
                next_path = path + [action]
                queue.push([successor_state, next_path])


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    # This problem is similar to Dijikstra's algorithm
    queue = util.PriorityQueue()
    queue.push([problem.getStartState(), [], 0], 0)
    visited_states = set()

    while True:
        current_state, path, state_cost = queue.pop()
        if problem.isGoalState(current_state):
            return path
        if not current_state in visited_states:
            visited_states.add(current_state)
            # Get successors information
            successors_tuple = problem.getSuccessors(current_state)
            # Add new state to stack
            for successor_state, action, cost in successors_tuple:
                if successor_state in visited_states:
                     continue
                next_path = path + [action]
                next_cost = state_cost + cost
                queue.push([successor_state, next_path, next_cost], next_cost)


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    queue = util.PriorityQueue()
    # F = g + h
    # Use F as the comparing value in the queue
    start_state = problem.getStartState()
    queue.push([start_state, [], 0], heuristic(start_state, problem) + 0)
    visited_states = set()

    while True:
        current_state, path, current_g = queue.pop()
        if problem.isGoalState(current_state):
            return path
        if not current_state in visited_states:
            visited_states.add(current_state)
            # Get successors information
            successors_tuple = problem.getSuccessors(current_state)
            # Add new state to stack
            for successor_state, action, cost in successors_tuple:
                if successor_state in visited_states:
                     continue
                next_path = path + [action]
                next_g = current_g + cost
                next_f = next_g + heuristic(successor_state, problem)
                queue.push([successor_state, next_path, next_g], next_f)


def biDirectionalAStarSearch(problem, heuristic):
    """
        The algorithm is actually same as Astar, the only difference is that we start from both start state and goal state. When both processes achieve to the same state, we can then know that we find a path from the start state to the goal by concatenating the path searched in both processes
    """
    def __reversedAction(actions):
        """
        Reversing the direction of all actions
        """
        return [Directions.REVERSE[x] for x in actions][::-1]

    # Initialiate necessary data structures for following algorithm
    pq1, pq2 = [PriorityQueue() for _ in range(2)]
    visited_states1, visited_states2 = [dict() for _ in range(2)]

    pq1.push((problem.getStartState(), [], 0), heuristic(problem.getStartState(), problem, ))
    pq2.push((problem.goal, [], 0), heuristic(problem.goal, problem, ))
    # Key is the state and the item stores all the actions to get the corresponding state
    visited_states1[problem.getStartState()] = []
    visited_states2[problem.goal] = []

    while not pq1.isEmpty() and not pq2.isEmpty():
        # Start process from the start state
        current_state, path, current_g = pq1.pop()

        # When there is a matching state, we then return the path
        if problem.isGoalState(current_state, visited_states2):
            return path + __reversedAction(visited_states2[current_state])

        for successor_state, action, cost in problem.getSuccessors(current_state):
            if successor_state in visited_states1:
                 continue
            next_path = path + [action]
            next_g = current_g + cost
            next_f = next_g + heuristic(successor_state, problem, 'goal')
            pq1.push((successor_state, next_path, next_g), next_f)
            visited_states1[successor_state] = next_path

        # Process from the goal state
        current_state, path, current_g = pq2.pop()
        if problem.isGoalState(current_state, visited_states1):
            return __reversedAction(visited_states1[current_state]) + path

        for successor_state, action, cost in problem.getSuccessors(current_state):
            if successor_state in visited_states2:
                 continue
            next_path = path + [action]
            next_g = current_g + cost
            next_f = next_g + heuristic(successor_state, problem, 'start')
            pq2.push((successor_state, next_path, next_g), next_f)
            visited_states2[successor_state] = next_path

    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
bdastar = biDirectionalAStarSearch
