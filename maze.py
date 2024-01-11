'''

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

'''

import argparse
import numpy as np

N,M = 10,10

parser = argparse.ArgumentParser()
parser.add_argument('--width', '-x', type=int, default=10, help='width of maze')
parser.add_argument('--heigh', '-y', type=int, default=10, help='height of maze')
parser.add_argument('--seed', '-s', type=int, default=None, help='random seed for maze generation (doesn\'t affect agent\' actions)')
parser.add_argument('--preset', '-p', type=int, default=2, help='Use preset maze, 0, 1, or 2.')
parser.add_argument('--random', '-r', action='store_true', help='If this is passed use random maze, otherwise use preset. WARNING: some mazes are impossible')
parser.add_argument('--verbose', '-v', action='store_true', help='Print details about agent progress')


args = parser.parse_args()
assert not args.random, 'random maze not yet supported'
assert 0<=args.preset<=2, 'only preset mazes 0, 1, and 2 are currently supported'

rng = np.random.default_rng(seed=args.seed)
quiet = not args.verbose
n = args.heigh
m = args.width

# ---------------------------------------------------

#  Generate Maze:

if args.random:
    # randomly generate maze with parameters defined in args
    board = rng.integers(0, 2, (n, m)) # fill randomly with 0s and 1s
    board[-1,-1]=-1
    print('Randomly Generated Maze:')

else:
    # use preset maze 0-2
    board = {
        0: np.array(
            [[0, 1, 1, 1, 0, 1, 0, 1, 1, 0],
             [0, 0, 1, 0, 1, 0, 0, 1, 1, 0],
             [0, 1, 1, 0, 0, 0, 1, 0, 0, 0],
             [0, 0, 1, 0, 1, 0, 1, 0, 0, 0],
             [0, 0, 0, 1, 0, 1, 1, 1, 0, 0],
             [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
             [1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
             [1, 1, 1, 1, 0, 1, 0, 1, 0, 1],
             [1, 1, 1, 1, 0, 0, 1, 0, 1, 0],
             [0, 0, 0, 0, 0, 0, 1, 1, 1, -1]]
            ),
        1: np.array(
        [[ 1,  1,  1,  1,  1,  0,  1,  1,  0,  0,],
         [ 0,  1,  1,  1,  0,  1,  1,  0,  0,  1,],
         [ 0,  0,  0,  1,  1,  1,  0,  1,  0,  0,],
         [ 0,  0,  0,  1,  0,  1,  1,  1,  0,  0,],
         [ 0,  1,  1,  0,  0,  0,  1,  1,  0,  0,],
         [ 1,  1,  0,  0,  1,  0,  0,  0,  1,  0,],
         [ 0,  0,  0,  1,  1,  0,  1,  1,  1,  1,],
         [ 0,  1,  1,  1,  0,  1,  0,  0,  0,  0,],
         [ 1,  1,  0,  1,  0,  0,  0,  1,  1,  1,],
         [ 0,  0,  0,  0,  0,  0,  1,  0,  0, -1,]]
        ),
        2: np.array(
        [[ 9,  0,  0,  0,  0,  0,  0,  0,  0,  0,],
         [ 0,  1,  1,  1,  0,  1,  1,  0,  0,  0,],
         [ 0,  0,  0,  1,  1,  1,  0,  1,  0,  0,],
         [ 0,  0,  0,  1,  0,  1,  1,  1,  0,  0,],
         [ 0,  1,  1,  0,  0,  0,  1,  1,  0,  0,],
         [ 0,  1,  0,  0,  1,  0,  0,  0,  1,  0,],
         [ 0,  0,  0,  1,  1,  0,  1,  1,  1,  0,],
         [ 0,  1,  1,  1,  0,  1,  0,  0,  0,  0,],
         [ 0,  1,  0,  1,  0,  0,  0,  1,  0,  0,],
         [ 0,  0,  0,  0,  0,  0,  0,  0,  0, -1,]]
        )
    }[args.preset]
    print('Premade maze')

print(board)

# ---------------------------------------------------

def sub_board(board,curr_c):
    # returns index range of 3x3 subgrid centered at agent's location
    y,x = curr_c

    if y==0: y_start=y
    else: y_start = y-1

    if x==0: x_start=x
    else: x_start=x-1

    if x==M-1: x_end=x
    else: x_end=x+1

    if y==N-1: y_end=y
    else: y_end=y+1

    return y_start,y_end,x_start,x_end


def prob_prop_to_value_episode(board, initial_c, Q, quiet, epsilon=0.01, dont_look_back=0.001, max_num_steps = 10**5):

    # Regret = np.zeros(board.shape)

    curr_c = initial_c # the agent starts at 0,0
    x_start, x_end, y_start, y_end = sub_board(board,curr_c) # this at-most 3x3 sub-array centered at curr_c.

    done = False
    steps = 1
    last_at_states = np.ones(Q.shape)*(-np.inf) # minus because will be subtracting from final steps.
    while ( (   not (-1 in board[y_start:y_end+1,x_start:x_end+1]))
                and  ( not done ) 
                and  ( steps<max_num_steps ) ):

        last_at_states[curr_c[0], curr_c[1]] = steps
        board[curr_c[0], curr_c[1]] = 9
        # boundaries of surroundings (inclusive so use end_z+1 in loops)

        if not quiet:
            print('\n\nsteps =',steps)
            print(('current coords '+str(curr_c)).center(80))
            print('board, state values'.center(80))
            print(board)
            print('state values:'.center(80))
            # print(Q)
            for line in Q-dont_look_back/(steps-last_at_states):
                print(line)
            print('surroundings:'.center(80))
            print(board[y_start:y_end+1,x_start:x_end+1])
            print()

        # best_adjacent_state = np.where(board[y_start:y_end+1,x_start:x_end+1]==np.max(np.max(board[y_start:y_end+1,x_start:x_end+1])))

        # Regret[curr_c[0],curr_c[1]]=0
        feasable_states = []
        transition_probs = []
        tmp_Q = Q-dont_look_back/(steps-last_at_states)
        for y in range(y_start, y_end+1):
            for x in range(x_start, x_end+1):
                if ( board[y, x]==0 ):
                    # according to the rules agent can only transfer to states with board value 0.
                    feasable_states.append((y, x))
                    transition_probs.append(tmp_Q[y, x])

        transition_probs = np.array(transition_probs).reshape(-1,) # required shape for np.random.choice().
        transition_probs = transition_probs*(transition_probs>0) + epsilon # take the max of 0, add epsilon.
        transition_probs = transition_probs/np.sum(transition_probs)
        next_idx = np.random.choice( list(range(transition_probs.size)), p = transition_probs )
        next_c = feasable_states[next_idx]

        if not quiet:
            print('feasable_states',feasable_states)
            print('greed',greed)
            print('curr_c',curr_c)
            print('next_c',next_c)

        board[curr_c[0], curr_c[1]] = 0
        board[next_c[0], next_c[1]] = 9

        curr_c=next_c
        y_start,y_end,x_start,x_end = sub_board(board,curr_c)

        if not quiet:
            exec(input('Done? (plz enter done=True or another command, or Enter to continue):'))

        steps += 1

    if steps<max_num_steps:
        last_at_states[curr_c[0],curr_c[1]] = steps
        last_at_states = (steps-last_at_states)
        np.save('last_at_states.npy', last_at_states)
        return True, steps

    else:
        return False, steps

def epsilon_greedy_episode(board, initial_c, Q, quiet, epsilon=0.01, dont_look_back=0.001, regret = 0.00001, max_num_steps = 10**5):

    # Regret = np.zeros(board.shape)

    curr_c = initial_c # the agent starts at 0,0
    x_start,x_end,y_start,y_end = sub_board(board,curr_c) # this at-most 3x3 sub-array centered at curr_c.

    done = False
    steps = 1
    last_at_states = np.ones(Q.shape)*(-np.inf) # minus because will be subtracting from final steps.
    while ( (   not (-1 in board[y_start:y_end+1,x_start:x_end+1])) 
                and  ( not done ) 
                and  ( steps<max_num_steps ) ):

        last_at_states[curr_c[0], curr_c[1]] = steps
        board[curr_c[0], curr_c[1]] = 9
        # boundaries of surroundings (inclusive so use end_z+1 in loops)

        if not quiet:
            print('\n\nsteps =',steps)
            print(('current coords '+str(curr_c)).center(80))
            print('board, state values'.center(80))
            print(board)
            print('state values:'.center(80))
            # print(Q)
            print(Q[y_start:y_end+1,x_start:x_end+1])
            # for line in Q-dont_look_back/(steps-last_at_states):
            #     print(line)
            print('surroundings:'.center(80))
            print(board[y_start:y_end+1,x_start:x_end+1])
            print()

        # best_adjacent_state = np.where(board[y_start:y_end+1,x_start:x_end+1]==np.max(np.max(board[y_start:y_end+1,x_start:x_end+1])))

        # Regret[curr_c[0],curr_c[1]]=0
        feasable_states = []
        for y in range(y_start,y_end+1):
            for x in range(x_start,x_end+1):
                if ( board[y,x]==0 ):
                    # according to the rules agent can only transfer to states with value 0.
                    feasable_states.append((y, x))

        np.random.shuffle(feasable_states) # in case all states are equally likely we don't want bias here.
        greed = np.random.rand()
        if greed < epsilon: # epsilon greedy random action for exploration,
            next_c = feasable_states[0] # it's shuffled

        else: # exploit current value estimate
            most_value = -np.inf # no state approx will ever be negative so this is overkill for being less than every Q[i,j].
            for y,x in feasable_states:
                q = Q[y,x]-dont_look_back/(steps-last_at_states[y,x]) # discourage re-visiting states.

                if q>most_value:
                    most_value = q
                    next_c = (y,x)

        if not quiet:
            print('feasable_states',feasable_states)
            print('greed',greed)
            print('curr_c',curr_c)
            print('next_c',next_c)

        board[curr_c[0],curr_c[1]] = 0
        board[next_c[0],next_c[1]] = 9
        curr_c=next_c
        y_start,y_end,x_start,x_end = sub_board(board,curr_c)

        if not quiet:
            exec(input('Done? (plz enter done=True or another command, or Enter to continue):'))

        steps += 1

    if steps<max_num_steps:
        last_at_states[curr_c[0],curr_c[1]] = steps
        last_at_states = (steps-last_at_states)
        np.save('last_at_states.npy',last_at_states)
        return True,steps

    else:
        return False,steps

if __name__=='__main__':

    # Learning rate parameters:
    gamma = 0.999
    alpha = 0.1

    # Policy hyperparameters:
    epsilon_greedy = True
    # epsilon_greedy = False
    # Policy parameters:
    board=board
    initial_c=(0, 0)
    epsilon=0.1
    dont_look_back=0.5

    # Q = np.load('Q.npy')
    Q = np.zeros(board.shape)

    # For exploration:

    # np.set_printoptions(precision=4)
    if epsilon_greedy:
        reached_goal, steps = epsilon_greedy_episode( board=board, initial_c=initial_c, Q=Q, quiet=quiet, epsilon=epsilon, dont_look_back=dont_look_back )
    else:
        reached_goal, steps = prob_prop_to_value_episode( board=board, initial_c=initial_c, Q=Q, quiet=quiet, epsilon=epsilon, dont_look_back=dont_look_back )

    if reached_goal:
        print('steps:', steps)
        with open('steps.txt', 'a') as file: 
            file.write(str(steps)+'\n')
        last_at_states=np.load('last_at_states.npy')
        print('board')
        print(board)
        print('path:')
        print(last_at_states)
        print('steps:',steps)
        Q_update = gamma**(last_at_states)
        # print('update\n',Q_update)
        Q = (1-alpha)*Q + alpha*Q_update
        np.save('Q.npy', Q)
        # print('Q\n',Q)

        import matplotlib.pyplot as plt
        import seaborn as sns; sns.set_theme()

        heat_map = sns.heatmap(Q)
        plt.show()

    else:
        print('Agent failed to find goal')
