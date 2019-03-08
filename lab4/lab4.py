# MIT 6.034 Lab 4: Constraint Satisfaction Problems
# Written by 6.034 staff

from constraint_api import *
from test_problems import get_pokemon_problem


#### Part 1: Warmup ############################################################

def has_empty_domains(csp) :
    """Returns True if the problem has one or more empty domains, otherwise False"""
    if [] in csp.domains.values():
        return True
    return False

def check_all_constraints(csp) :
    """Return False if the problem's assigned values violate some constraint,
    otherwise True"""
    for var1, val1 in csp.assignments.items():
        for var2, val2 in csp.assignments.items():
            for constraint in csp.constraints_between(var1, var2):
                if not constraint.check(val1, val2):
                    return False
    return True


#### Part 2: Depth-First Constraint Solver #####################################

def solve_constraint_dfs(problem) :
    """
    Solves the problem using depth-first search.  Returns a tuple containing:
    1. the solution (a dictionary mapping variables to assigned values)
    2. the number of extensions made (the number of problems popped off the agenda).
    If no solution was found, return None as the first element of the tuple.
    """
    agenda = [problem]
    extensions = 0
    solution = None

    while not len(agenda) == 0:
        current = agenda[0].copy()
        del agenda[0]
        extensions += 1
        if not (has_empty_domains(current) or not check_all_constraints(current)):
            if current.unassigned_vars:
                var = current.pop_next_unassigned_var()
                new_problems = []
                for val in current.get_domain(var):
                    new_problems.append(current.copy().set_assignment(var, val))
                new_problems.reverse()
                for problem in new_problems:
                    agenda.insert(0,problem)
            else:
                solution = current.assignments
                break

    return (solution, extensions)
        

# QUESTION 1: How many extensions does it take to solve the Pokemon problem
#    with DFS?

# Hint: Use get_pokemon_problem() to get a new copy of the Pokemon problem
#    each time you want to solve it with a different search method.

ANSWER_1 = 20


#### Part 3: Forward Checking ##################################################

def eliminate_from_neighbors(csp, var) :
    """
    Eliminates incompatible values from var's neighbors' domains, modifying
    the original csp.  Returns an alphabetically sorted list of the neighboring
    variables whose domains were reduced, with each variable appearing at most
    once.  If no domains were reduced, returns empty list.
    If a domain is reduced to size 0, quits immediately and returns None.
    """
    def constraint_check(var1, var2, val1, val2):
        for constraint in csp.constraints_between(var1, var2):
            if not constraint.check(val1, val2):
                return True
        return False

    domains = []
    for neigh in csp.get_neighbors(var):
        con_values = []
        for nVal in csp.get_domain(neigh):
            constraint_violated = True

            for val in csp.get_domain(var):
                if not constraint_check(var, neigh, val, nVal):
                    constraint_violated = False

            if constraint_violated:                
                domains.append(neigh)
                con_values.append(nVal)
        for value in con_values:
            csp.eliminate(neigh, value)
            if len(csp.get_domain(neigh)) == 0:
                return None

    return sorted(list(set(domains)))

# Because names give us power over things (you're free to use this alias)
forward_check = eliminate_from_neighbors

def solve_constraint_forward_checking(problem) :
    """
    Solves the problem using depth-first search with forward checking.
    Same return type as solve_constraint_dfs.
    """

    agenda = [problem]
    extensions = 0
    solution = None

    while not len(agenda) == 0:
        current = agenda[0].copy()
        del agenda[0]
        extensions += 1
        if not (has_empty_domains(current) or not check_all_constraints(current)):
            if current.unassigned_vars:
                var = current.pop_next_unassigned_var()
                new_problems = []
                for val in current.get_domain(var):
                    new_problems.append(current.copy().set_assignment(var, val))
                new_problems.reverse()
                for problem in new_problems:
                    forward_check(problem, var)
                    agenda.insert(0,problem)
            else:
                solution = current.assignments
                break

    return (solution, extensions)

# QUESTION 2: How many extensions does it take to solve the Pokemon problem
#    with DFS and forward checking?

ANSWER_2 = 9


#### Part 4: Domain Reduction ##################################################

def domain_reduction(csp, queue=None) :
    """
    Uses constraints to reduce domains, propagating the domain reduction
    to all neighbors whose domains are reduced during the process.
    If queue is None, initializes propagation queue by adding all variables in
    their default order. 
    Returns a list of all variables that were dequeued, in the order they
    were removed from the queue.  Variables may appear in the list multiple times.
    If a domain is reduced to size 0, quits immediately and returns None.
    This function modifies the original csp.
    """
    if not queue:
        queue = csp.get_all_variables()
    dequeued = []

    while not len(queue) == 0:
        current = queue[0]
        dequeued.append(current)
        del queue[0]

        values = eliminate_from_neighbors(csp,current)
        if values is None:
            return None
        else:
            for value in values:
                queue.append(value)

    return dequeued

# QUESTION 3: How many extensions does it take to solve the Pokemon problem
#    with DFS (no forward checking) if you do domain reduction before solving it?

ANSWER_3 = 6


def solve_constraint_propagate_reduced_domains(problem) :
    """
    Solves the problem using depth-first search with forward checking and
    propagation through all reduced domains.  Same return type as
    solve_constraint_dfs.
    """
    agenda = [problem]
    extensions = 0
    solution = None

    while not len(agenda) == 0:
        current = agenda[0].copy()
        del agenda[0]
        extensions += 1
        if not (has_empty_domains(current) or not check_all_constraints(current)):
            if current.unassigned_vars:
                var = current.pop_next_unassigned_var()
                new_problems = []
                for val in current.get_domain(var):
                    new_prob = current.copy().set_assignment(var, val)
                    domain_reduction(new_prob)
                    new_problems.append(new_prob)
                new_problems.reverse()
                for problem in new_problems:
                    forward_check(problem, var)
                    agenda.insert(0,problem)
            else:
                solution = current.assignments
                break
    return (solution, extensions)


# QUESTION 4: How many extensions does it take to solve the Pokemon problem
#    with forward checking and propagation through reduced domains?

ANSWER_4 = 7


#### Part 5A: Generic Domain Reduction #########################################

def propagate(enqueue_condition_fn, csp, queue=None) :
    """
    Uses constraints to reduce domains, modifying the original csp.
    Uses enqueue_condition_fn to determine whether to enqueue a variable whose
    domain has been reduced. Same return type as domain_reduction.
    """
    if not queue:
        queue = csp.get_all_variables()
    dequeued = []

    while not len(queue) == 0:
        current = queue[0]
        dequeued.append(current)
        del queue[0]

        values = eliminate_from_neighbors(csp,current)
        if values is None:
            return None
        else:
            for value in values:
                if enqueue_condition_fn(csp,value):
                    queue.append(value)

    return dequeued

def condition_domain_reduction(csp, var) :
    """Returns True if var should be enqueued under the all-reduced-domains
    condition, otherwise False"""

    return True

def condition_singleton(csp, var) :
    """Returns True if var should be enqueued under the singleton-domains
    condition, otherwise False"""
    if len(csp.get_domain(var)) == 1:
        return True
    return False

def condition_forward_checking(csp, var) :
    """Returns True if var should be enqueued under the forward-checking
    condition, otherwise False"""
    return False


#### Part 5B: Generic Constraint Solver ########################################

def solve_constraint_generic(problem, enqueue_condition=None) :
    """
    Solves the problem, calling propagate with the specified enqueue
    condition (a function). If enqueue_condition is None, uses DFS only.
    Same return type as solve_constraint_dfs.
    """
    agenda = [problem]
    extensions = 0
    solution = None

    while not len(agenda) == 0:
        current = agenda[0].copy()
        del agenda[0]
        extensions += 1
        if not (has_empty_domains(current) or not check_all_constraints(current)):
            if current.unassigned_vars:
                var = current.pop_next_unassigned_var()
                new_problems = []
                for val in current.get_domain(var):
                    new_prob = current.copy().set_assignment(var, val)
                    if enqueue_condition:
                        propagate(enqueue_condition,new_prob)
                    new_problems.append(new_prob)
                new_problems.reverse()
                for problem in new_problems:
                    forward_check(problem, var)
                    agenda.insert(0,problem)
            else:
                solution = current.assignments
                break
    return (solution, extensions)

# QUESTION 5: How many extensions does it take to solve the Pokemon problem
#    with forward checking and propagation through singleton domains? (Don't
#    use domain reduction before solving it.)
poke = get_pokemon_problem()
print(solve_constraint_generic(poke,condition_forward_checking))
ANSWER_5 = 7


#### Part 6: Defining Custom Constraints #######################################

def constraint_adjacent(m, n) :
    """Returns True if m and n are adjacent, otherwise False.
    Assume m and n are ints."""
    if abs(m-n) == 1:
        return True
    return False

def constraint_not_adjacent(m, n) :
    """Returns True if m and n are NOT adjacent, otherwise False.
    Assume m and n are ints."""
    return not constraint_adjacent(m, n)

def all_different(variables) :
    """Returns a list of constraints, with one difference constraint between
    each pair of variables."""
    constraints = []
    for var1 in variables:
        for var2 in variables:
            if var1 != var2 and Constraint(var2,var1, constraint_different) not in constraints:
                constraints.append(Constraint(var1,var2, constraint_different))
    return constraints



#### SURVEY ####################################################################

NAME = "Katya Bezugla"
COLLABORATORS = "None"
HOW_MANY_HOURS_THIS_LAB_TOOK = 12
WHAT_I_FOUND_INTERESTING = "creating the helper functions to make my code simpler"
WHAT_I_FOUND_BORING = "None"
SUGGESTIONS = "None"
