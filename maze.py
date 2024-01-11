'''

Author: Seth Briney

Credit: Barto, Sutton, Adam White, Martha White, Richard Weiss

Write policy iteration type algorithm using numpy arrays.

Everything will be numpy arrays.

prob_prop_to_value_episode makes decisions proportional to the running estimate of the state value to transition to.

epsilon_greedy_episode is as described, except for the following:

both prob_prop_to_value_episode and epsilon_greedy_episode have a offset to discourage backtracking, that falls off by exactly 1/(steps-last_at_states[y,x]) where y,x is the state in consideration.

'''

import numpy as np
N,M = 10,10

# ---------------------------------------------------

#  For generating stuff.

# def rand_binary(n,m): # generate random binary nxm array.
#     return ( np.random.randn(n,m)>0 ).astype(int)

# board = rand_binary(N,M)
# board[-1,-1]=-1
# print(board)

# board = np.zeros((N,M))
# print(board)

# ---------------------------------------------------

# board = np.array( # randomly generated sort of maze.
#     [[0, 1, 1, 1, 0, 1, 0, 1, 1, 0],
#      [0, 0, 1, 0, 1, 0, 0, 1, 1, 0],
#      [0, 1, 1, 0, 0, 0, 1, 0, 0, 0],
#      [0, 0, 1, 0, 1, 0, 1, 0, 0, 0],
#      [0, 0, 0, 1, 0, 1, 1, 1, 0, 0],
#      [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
#      [1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
#      [1, 1, 1, 1, 0, 1, 0, 1, 0, 1],
#      [1, 1, 1, 1, 0, 0, 1, 0, 1, 0],
#      [0, 0, 0, 0, 0, 0, 1, 1, 1, -1]]
#      )
# ---------------------------------------------------

# board = np.array( # randomly generated sort of maze.
# [[ 1,  1,  1,  1,  1,  0,  1,  1,  0,  0,],
#  [ 0,  1,  1,  1,  0,  1,  1,  0,  0,  1,],
#  [ 0,  0,  0,  1,  1,  1,  0,  1,  0,  0,],
#  [ 0,  0,  0,  1,  0,  1,  1,  1,  0,  0,],
#  [ 0,  1,  1,  0,  0,  0,  1,  1,  0,  0,],
#  [ 1,  1,  0,  0,  1,  0,  0,  0,  1,  0,],
#  [ 0,  0,  0,  1,  1,  0,  1,  1,  1,  1,],
#  [ 0,  1,  1,  1,  0,  1,  0,  0,  0,  0,],
#  [ 1,  1,  0,  1,  0,  0,  0,  1,  1,  1,],
#  [ 0,  0,  0,  0,  0,  0,  1,  0,  0, -1,]]
# )
# ---------------------------------------------------

board = np.array( # randomly generated sort of maze.
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

# ---------------------------------------------------

def sub_board(board,curr_c):
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

# The idea here is that an agent (represented by 9) will move through this array (board) until it reaches the terminal state -1.

# A state is a pair of indeces n,m which represents a place on the board.

# Rules:

# The agent can only occupy states with 0,
# The agent can make jump to any 0 entry within the 3x3 sub-array centered at the current state.
# The terminal state is -1, and an episode will end once the terminal state is contained in the 3x3 sub-array centered at the state. 

# I will use "the surroundings" to refer to "3x3 sub-array centered at the state".
def prob_prop_to_value_episode(board, initial_c, Q, quiet, epsilon=0.01, dont_look_back=0.001, max_num_steps = 10**5):

    # Regret = np.zeros(board.shape)

    curr_c = initial_c # the agent starts at 0,0
    x_start,x_end,y_start,y_end = sub_board(board,curr_c) # this at-most 3x3 sub-array centered at curr_c.

    done = False
    steps = 1
    last_at_states = np.ones(Q.shape)*(-np.inf) # minus because will be subtracting from final steps.
    while ( 
    (not (-1 in board[y_start:y_end+1,x_start:x_end+1])) 
     and  ( not done ) 
     and  ( steps<max_num_steps ) ):

        last_at_states[curr_c[0],curr_c[1]] = steps
        board[curr_c[0],curr_c[1]] = 9
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
        for y in range(y_start,y_end+1):
            for x in range(x_start,x_end+1):
                if ( board[y,x]==0 ):
                    # according to the rules agent can only transfer to states with value 0.
                    feasable_states.append((y,x))
                    transition_probs.append(tmp_Q[y,x])
        transition_probs=np.array(transition_probs).reshape(-1,) # required shape for np.random.choice().
        transition_probs=transition_probs*(transition_probs>0)+epsilon # take the max of 0, add epsilon.
        transition_probs = transition_probs/np.sum(transition_probs)
        next_c = feasable_states[
                np.random.choice( 
                    [n for n in range(transition_probs.size)]
                    , p = transition_probs ) ]
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

def epsilon_greedy_episode(board, initial_c, Q, quiet, epsilon=0.01, dont_look_back=0.001, regret = 0.00001, max_num_steps = 10**5):

    # Regret = np.zeros(board.shape)

    curr_c = initial_c # the agent starts at 0,0
    x_start,x_end,y_start,y_end = sub_board(board,curr_c) # this at-most 3x3 sub-array centered at curr_c.

    done = False
    steps = 1
    last_at_states = np.ones(Q.shape)*(-np.inf) # minus because will be subtracting from final steps.
    while ( 
    (not (-1 in board[y_start:y_end+1,x_start:x_end+1])) 
     and  ( not done ) 
     and  ( steps<max_num_steps ) ):

        last_at_states[curr_c[0],curr_c[1]] = steps
        board[curr_c[0],curr_c[1]] = 9
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
                    feasable_states.append((y,x))
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
    initial_c=(0,0)
    quiet=True
    epsilon=0.1
    dont_look_back=0.5

    # Q = np.load('Q.npy')
    Q = np.zeros(board.shape)

    # For exploration:

    # np.set_printoptions(precision=4)
    if epsilon_greedy:
        reached_goal, steps = epsilon_greedy_episode( board=board,initial_c=initial_c,Q=Q,quiet=quiet,epsilon=epsilon,dont_look_back=dont_look_back )
    else:
        reached_goal, steps = prob_prop_to_value_episode( board=board,initial_c=initial_c,Q=Q,quiet=quiet,epsilon=epsilon,dont_look_back=dont_look_back )


    if reached_goal:
        print('steps:',steps)
        with open('steps.txt','a') as file: 
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
        np.save('Q.npy',Q)
        # print('Q\n',Q)

        import matplotlib.pyplot as plt
        import seaborn as sns; sns.set_theme()

        heat_map = sns.heatmap(Q)
        plt.show()
