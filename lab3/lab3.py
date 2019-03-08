# MIT 6.034 Lab 3: Games
# Written by 6.034 staff

from game_api import *
from boards import *
from toytree import GAME1

INF = float('inf')

# Please see wiki lab page for full description of functions and API.

#### Part 1: Utility Functions #################################################

def is_game_over_connectfour(board):
    """Returns True if game is over, otherwise False."""
    for chain in board.get_all_chains():
        if len(chain) >= 4:
            return True
    for col in range(board.num_cols):
        if not board.is_column_full(col):
            return False
    return True

def next_boards_connectfour(board):
    """Returns a list of ConnectFourBoard objects that could result from the
    next move, or an empty list if no moves can be made."""
    new_boards = []
    if not is_game_over_connectfour(board):
        for col in range(board.num_cols):
            if not board.is_column_full(col):
                new_boards.append(board.add_piece(col))
    return new_boards

def endgame_score_connectfour(board, is_current_player_maximizer):
    """Given an endgame board, returns 1000 if the maximizer has won,
    -1000 if the minimizer has won, or 0 in case of a tie."""
    for chain in board.get_all_chains(is_current_player_maximizer):
        if len(chain) >= 4:
            return 1000
    for chain in board.get_all_chains(not is_current_player_maximizer):
        if len(chain) >= 4:
            return -1000
    return 0

def endgame_score_connectfour_faster(board, is_current_player_maximizer):
    """Given an endgame board, returns an endgame score with abs(score) >= 1000,
    returning larger absolute scores for winning sooner."""
    return endgame_score_connectfour(board, is_current_player_maximizer) * 1000/board.count_pieces()

def heuristic_connectfour(board, is_current_player_maximizer):
    """Given a non-endgame board, returns a heuristic score with
    abs(score) < 1000, where higher numbers indicate that the board is better
    for the maximizer."""
    len_curr = 0
    len_opp = 0
    for chain in board.get_all_chains(is_current_player_maximizer):
        len_curr += len(chain)
        if len(chain) >= 3:
            len_curr += len(chain)
    for chain in board.get_all_chains(not is_current_player_maximizer):
        len_opp += len(chain)
        if len(chain) >= 3:
            len_opp += len(chain)
    return len_curr - len_opp

# Now we can create AbstractGameState objects for Connect Four, using some of
# the functions you implemented above.  You can use the following examples to
# test your dfs and minimax implementations in Part 2.

# This AbstractGameState represents a new ConnectFourBoard, before the game has started:
state_starting_connectfour = AbstractGameState(snapshot = ConnectFourBoard(),
                                 is_game_over_fn = is_game_over_connectfour,
                                 generate_next_states_fn = next_boards_connectfour,
                                 endgame_score_fn = endgame_score_connectfour_faster)

# This AbstractGameState represents the ConnectFourBoard "NEARLY_OVER" from boards.py:
state_NEARLY_OVER = AbstractGameState(snapshot = NEARLY_OVER,
                                 is_game_over_fn = is_game_over_connectfour,
                                 generate_next_states_fn = next_boards_connectfour,
                                 endgame_score_fn = endgame_score_connectfour_faster)

# This AbstractGameState represents the ConnectFourBoard "BOARD_UHOH" from boards.py:
state_UHOH = AbstractGameState(snapshot = BOARD_UHOH,
                                 is_game_over_fn = is_game_over_connectfour,
                                 generate_next_states_fn = next_boards_connectfour,
                                 endgame_score_fn = endgame_score_connectfour_faster)


#### Part 2: Searching a Game Tree #############################################

# Note: Functions in Part 2 use the AbstractGameState API, not ConnectFourBoard.

def dfs_maximizing(state) :
    """Performs depth-first search to find path with highest endgame score.
    Returns a tuple containing:
     0. the best path (a list of AbstractGameState objects),
     1. the score of the leaf node (a number), and
     2. the number of static evaluations performed (a number)"""
    result = [state, -INF, 0]
    queue = [[state]]
    while not len(queue) == 0:
        current = queue[0].copy()
        del queue[0]
        if current[-1].is_game_over():
            result[2] += 1
            if result[1] < current[-1].get_endgame_score():
                result[0] = current
                result[1] = current[-1].get_endgame_score()
        else:
            for s in current[-1].generate_next_states():
                new_path = current.copy()
                new_path.append(s)
                queue.insert(0, new_path)
    return (result[0], result[1], result[2])


# Uncomment the line below to try your dfs_maximizing on an
# AbstractGameState representing the games tree "GAME1" from toytree.py:

#pretty_print_dfs_type(dfs_maximizing(GAME1))


def minimax_endgame_search(state, maximize=True) :
    """Performs minimax search, searching all leaf nodes and statically
    evaluating all endgame scores.  Same return type as dfs_maximizing."""
    def return_score(e):
        return e[1]
    if state.is_game_over():
        #print(state.get_endgame_score(maximize))
        return ([state], state.get_endgame_score(maximize), 1)
    else:
        result = []
        for s in state.generate_next_states():
            result.append(minimax_endgame_search(s, not maximize))
        m_result = 0
        if maximize:
            m_result = max(result,key=return_score)
        else:
            m_result = min(result,key=return_score)
        sum_result = 0
        for r in result:
            sum_result += r[2]
        m_result[0].insert(0,state)
        #print(m_result)
        return (m_result[0], m_result[1], sum_result)




# Uncomment the line below to try your minimax_endgame_search on an
# AbstractGameState representing the ConnectFourBoard "NEARLY_OVER" from boards.py:

# pretty_print_dfs_type(minimax_endgame_search(state_NEARLY_OVER))


def minimax_search(state, heuristic_fn=always_zero, depth_limit=INF, maximize=True) :
    """Performs standard minimax search. Same return type as dfs_maximizing."""
    def return_score(e):
        return e[1]
    if state.is_game_over():
        #print(state.get_endgame_score(maximize))
        return ([state], state.get_endgame_score(maximize), 1)
    elif depth_limit == 0:
        return ([state], heuristic_fn(state.get_snapshot(), maximize), 1)
    else:
        result = []
        for s in state.generate_next_states():
            result.append(minimax_search(s, heuristic_fn, depth_limit - 1, not maximize))
        m_result = 0
        if maximize:
            m_result = max(result,key=return_score)
        else:
            m_result = min(result,key=return_score)
        sum_result = 0
        for r in result:
            sum_result += r[2]
        m_result[0].insert(0,state)
        #print(m_result)
        return (m_result[0], m_result[1], sum_result)


# Uncomment the line below to try minimax_search with "BOARD_UHOH" and
# depth_limit=1. Try increasing the value of depth_limit to see what happens:

# pretty_print_dfs_type(minimax_search(state_UHOH, heuristic_fn=heuristic_connectfour, depth_limit=1))


def minimax_search_alphabeta(state, alpha=-INF, beta=INF, heuristic_fn=always_zero,
                             depth_limit=INF, maximize=True) :
    """"Performs minimax with alpha-beta pruning. Same return type 
    as dfs_maximizing."""
    if state.is_game_over():
        return ([state], state.get_endgame_score(maximize), 1)
    elif depth_limit == 0:
        return ([state], heuristic_fn(state.get_snapshot(), maximize), 1)
    else:
        result_value = 0
        result_node = 0
        result_sum = 0
        alpha_node = [state]
        beta_node = [state]
        for s in state.generate_next_states():
            result = minimax_search_alphabeta(s, alpha, beta, heuristic_fn, depth_limit - 1, not maximize)
            result_sum += result[2]
            result_value = result[1]
            result_node = result[0]
            if maximize:
                if result_value > alpha:
                    alpha = result_value
                    alpha_node = result_node
            else:
                if result_value < beta:
                    beta = result_value
                    beta_node = result_node
            
            if alpha >= beta:
                break
        if maximize:
            alpha_node.insert(0,state)
            return (alpha_node, alpha, result_sum)
        else:
            beta_node.insert(0,state)
            return (beta_node, beta, result_sum)



# Uncomment the line below to try minimax_search_alphabeta with "BOARD_UHOH" and
# depth_limit=4. Compare with the number of evaluations from minimax_search for
# different values of depth_limit.

# pretty_print_dfs_type(minimax_search_alphabeta(state_UHOH, heuristic_fn=heuristic_connectfour, depth_limit=4))


def progressive_deepening(state, heuristic_fn=always_zero, depth_limit=INF,
                          maximize=True) :
    """Runs minimax with alpha-beta pruning. At each level, updates anytime_value
    with the tuple returned from minimax_search_alphabeta. Returns anytime_value."""
    anytime_value = AnytimeValue()
    for d in range(1, depth_limit+1):
        anytime_value.set_value(minimax_search_alphabeta(state, -INF, INF, heuristic_fn, d, maximize))
    return anytime_value


# Uncomment the line below to try progressive_deepening with "BOARD_UHOH" and
# depth_limit=4. Compare the total number of evaluations with the number of
# evaluations from minimax_search or minimax_search_alphabeta.

# progressive_deepening(state_UHOH, heuristic_fn=heuristic_connectfour, depth_limit=4).pretty_print()


# Progressive deepening is NOT optional. However, you may find that 
#  the tests for progressive deepening take a long time. If you would
#  like to temporarily bypass them, set this variable False. You will,
#  of course, need to set this back to True to pass all of the local
#  and online tests.
TEST_PROGRESSIVE_DEEPENING = True
if not TEST_PROGRESSIVE_DEEPENING:
    def not_implemented(*args): raise NotImplementedError
    progressive_deepening = not_implemented


#### Part 3: Multiple Choice ###################################################

ANSWER_1 = '4'

ANSWER_2 = '1'

ANSWER_3 = '4'

ANSWER_4 = '5'


#### SURVEY ###################################################

NAME = "Katya Bezugla"
COLLABORATORS = "None"
HOW_MANY_HOURS_THIS_LAB_TOOK = 7
WHAT_I_FOUND_INTERESTING = "Writing the min max algorithms"
WHAT_I_FOUND_BORING = "None"
SUGGESTIONS = None
