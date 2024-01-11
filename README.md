This is a slight revision of a maze-running reinforcement learning AI I wrote while taking some undergrad classes at The Evergreen State College, in preparation for my Master of Computer Science degree.

It produces a graphical output of the Q values of the maze grid.

Author: Seth Briney

Credit: Barto, Sutton, Adam White, Martha White, Richard Weiss

Assignment: Write policy iteration type algorithm using numpy arrays.

The following program utilizes tabular Q-learning to train a maze-runner reinforcement learning control agent.

prob_prop_to_value_episode makes decisions proportional to the running estimate of the state value to transition to.

epsilon_greedy_episode is as described, except for the following:

both prob_prop_to_value_episode and epsilon_greedy_episode have an offset to discourage backtracking, that falls off by exactly 1/(steps-last_at_states[y,x]) where y,x is the state in consideration.

The idea here is that an agent (represented by 9) will move through this array (board) until it reaches the terminal state -1.

A state is a pair of indeces n,m which represents a place on the board.

Rules:

The agent can only occupy states with 0,
The agent can make jump to any 0 entry within the 3x3 sub-array centered at the current state.
The terminal state is -1, and an episode will end once the terminal state is contained in the 3x3 sub-array centered at the state. 

I will use "the surroundings" to refer to "3x3 sub-array centered at the state".
