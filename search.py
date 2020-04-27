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
from game import reverseActions
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
                next_f = next_g + heuristic(successor_state, problem, "goal")
                queue.push([successor_state, next_path, next_g], next_f)

def biDirectionalAStarSearch(problem, heuristic):
    """
        The algorithm is actually same as Astar, the only difference is that we start from both start state and goal state. When both processes achieve to the same state, we can then know that we find a path from the start state to the goal by concatenating the path searched in both processes
    """
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
            return path + reverseActions(visited_states2[current_state])

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
            return visited_states1[current_state] + reverseActions(path)

        for successor_state, action, cost in problem.getSuccessors(current_state):
            if successor_state in visited_states2:
                 continue
            next_path = path + [action]
            next_g = current_g + cost
            next_f = next_g + heuristic(successor_state, problem, 'start')
            pq2.push((successor_state, next_path, next_g), next_f)
            visited_states2[successor_state] = next_path

    return []

def biDirectionalBFSSearch(problem):
    q1, q2 = [list() for _ in range(2)]
    visited1, visited2 = [dict() for _ in range(2)]

    q1.append((problem.getStartState(), []))
    q2.append((problem.goal, []))
    visited1[problem.getStartState()] = []
    visited2[problem.goal] = []

    while len(q1) and len(q2):
        # Progress starts from the start state
        new_queue = []
        for state, path in q1:
            if state in visited2:
                return path + reverseActions(visited2[state])

            for successor_state, action, cost in problem.getSuccessors(state):
                if successor_state in visited1:
                     continue
                next_path = path + [action]
                new_queue.append([successor_state, next_path])
                visited1[successor_state] = next_path
        q1 = new_queue
        new_queue = []

        # Progress starts from the goal state
        for state, path in q2:
            if state in visited1:
                return visited1[state] + reverseActions(path)

            for successor_state, action, cost in problem.getSuccessors(state):
                if successor_state in visited2:
                     continue
                next_path = path + [action]
                new_queue.append([successor_state, next_path])
                visited2[successor_state] = next_path
        q2 = new_queue

    return []

def biDirectionalMMSearch(problem, heuristic):
    """
       Implemented following the pseudo code from:
       R. Holte, A. Felner, G. Sharon, and N. Sturtevant. "Bidirectional Search That 
       Is Guaranteed to Meet in the Middle". 
    """
    def priority(f, g):
        return max(f, 2 * g)

    def get_min(open_list, g, end, prob):
        # Find minimum priority g and f values
        pr_min, min_f, min_g = float('inf'), float('inf'), float('inf')
        for s in open_list:
            pr_min = min(pr_min, g[s][0])
            f = g[s][2]
            min_f = min(min_f, f)
            min_g = min(min_g, g[s][1])

        return pr_min, min_f, min_g

    def find_next_node(pr_min, open_list, g):
        # Find next node - must satisfy having lowest priority + g value
        i = float('inf')
        node = open_list[0]
        for s in open_list:
            pri = g[s][0]
            g_val = g[s][1]
            if pri == pr_min:
                if g_val < i:
                    i = g_val
                    node = s

        return node

    def expand(U, g_this, g_other, open_this, open_other, closed_this, end, prob):
        # Expand search tree
        node = find_next_node(C, open_this, g_this)

        open_this.remove(node)
        closed_this.append(node)

        for succ_state, action, cost in prob.getSuccessors(node):
            curr_cost = g_this[node][1]
            next_cost = curr_cost + cost

            if succ_state in open_this or succ_state in closed_this:  
                if cost <= next_cost:
                    continue

                open_this.remove(succ_state)
                # Seems this part is not necessary in practice
                # Removed to conserve cycles
                # closed_this.remove(succ_state)

            curr_path = g_this[node][3]
            next_path = curr_path + [action]
            heur = heuristic(succ_state, problem, end)
            next_f = next_cost + heur

            g_this[succ_state] = (priority(next_f, next_cost), next_cost, next_f, next_path)
            open_this.append(succ_state)

            if succ_state in open_other:
                U = min(U, g_this[succ_state][1] + g_other[succ_state][1])

        return U, g_this, open_this, closed_this

    openF, openB = [problem.getStartState()], [problem.goal]
    closedF, closedB = [], []
    
    # {state: priority, g, f, path history}
    gF, gB = {problem.getStartState(): (0, 0, 0, [])}, {problem.goal: (0, 0, 0, [])}

    U = float('inf')
    e = 1

    while not len(openF)==0 and not len(openB)==0:

        prMinF, f_min_f, g_min_f = get_min(openF, gF, "goal", problem)
        prMinB, f_min_b, g_min_b = get_min(openB, gB, "start", problem)

        C = min(prMinF, prMinB)
      
        if U <= max(C, f_min_f, f_min_b, g_min_f + g_min_b + e):
            for state in openB:
                if state in openF:
                    return gF[state][3] + reverseActions(gB[state][3])

        if C == prMinF:
            # Forward
            U, gF, openF, closedF = expand(U, gF, gB, openF, openB, closedF, "goal", problem)
        else:
            # Backward
            U, gB, openB, closedB = expand(U, gB, gF, openB, openF, closedB, "start", problem)

    return []

def biDirectionalMM0Search(problem):
    """
       Implemented following the pseudo code from:
       R. Holte, A. Felner, G. Sharon, and N. Sturtevant. "Bidirectional Search That 
       Is Guaranteed to Meet in the Middle". 
    """
    def priority(g):
        return 2 * g

    def get_min(open_list, g):
        # Find minimum priority g value
        pr_min, min_g = float('inf'), float('inf')
        for s in open_list:
            pr_min = min(pr_min, g[s][0])
            min_g = min(min_g, g[s][1])

        return pr_min, min_g

    def find_next_node(pr_min, open_list, g):
        # Find next node - must satisfy having lowest priority + g value
        i = float('inf')
        node = open_list[0]
        for s in open_list:
            pri = g[s][0]
            g_val = g[s][1]
            if pri == pr_min:
                if g_val < i:
                    i = g_val
                    node = s

        return node

    def expand(U, g_this, g_other, open_this, open_other, closed_this, prob):
        # Expand search tree
        node = find_next_node(C, open_this, g_this)

        open_this.remove(node)
        closed_this.append(node)

        for succ_state, action, cost in prob.getSuccessors(node):
            curr_cost = g_this[node][1]
            next_cost = curr_cost + cost

            if succ_state in open_this or succ_state in closed_this:  
                if cost <= next_cost:
                    continue

                open_this.remove(succ_state)
                # Seems this part is not necessary in practice
                # Removed to conserve cycles
                # closed_this.remove(succ_state)

            curr_path = g_this[node][2]
            next_path = curr_path + [action]

            g_this[succ_state] = (priority(next_cost), next_cost, next_path)
            open_this.append(succ_state)

            if succ_state in open_other:
                U = min(U, g_this[succ_state][1] + g_other[succ_state][1])

        return U, g_this, open_this, closed_this

    openF, openB = [problem.getStartState()], [problem.goal]
    closedF, closedB = [], []
    
    # {state: priority, g, path history}
    gF, gB = {problem.getStartState(): (0, 0, [])}, {problem.goal: (0, 0, [])}

    U = float('inf')
    e = 1

    while not len(openF)==0 and not len(openB)==0:

        prMinF, g_min_f = get_min(openF, gF)
        prMinB, g_min_b = get_min(openB, gB)

        C = min(prMinF, prMinB)
      
        if U <= max(C, g_min_f + g_min_b + e):
            for state in openB:
                if state in openF:
                    return gF[state][2] + reverseActions(gB[state][2])

        if C == prMinF:
            # Forward
            U, gF, openF, closedF = expand(U, gF, gB, openF, openB, closedF, problem)
        else:
            # Backward
            U, gB, openB, closedB = expand(U, gB, gF, openB, openF, closedB, problem)

    return []

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
bdastar = biDirectionalAStarSearch
bdbfs = biDirectionalBFSSearch
bdmms = biDirectionalMMSearch
bdmm0 = biDirectionalMM0Search
