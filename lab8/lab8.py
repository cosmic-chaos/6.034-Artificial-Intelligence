# MIT 6.034 Lab 8: Bayesian Inference
# Written by 6.034 staff

from nets import *


#### Part 1: Warm-up; Ancestors, Descendents, and Non-descendents ##############

def get_ancestors(net, var):
    "Return a set containing the ancestors of var"
    ancestors = []
    variable = var
    for parent in net.get_parents(var):
        ancestors.append(parent)
        ancestors += get_ancestors(net, parent)

    return set(ancestors)

def get_descendants(net, var):
    "Returns a set containing the descendants of var"
    descendants = []
    variable = var
    for child in net.get_children(var):
        descendants.append(child)
        descendants += get_descendants(net, child)

    return set(descendants)

def get_nondescendants(net, var):
    "Returns a set containing the non-descendants of var"
    descendants = get_descendants(net,var)
    nondescendants = []
    for variable in net.get_variables():
        if variable not in descendants and variable != var:
            nondescendants.append(variable)
    return set(nondescendants)


#### Part 2: Computing Probability #############################################

def simplify_givens(net, var, givens):
    """
    If givens include every parent of var and no descendants, returns a
    simplified list of givens, keeping only parents.  Does not modify original
    givens.  Otherwise, if not all parents are given, or if a descendant is
    given, returns original givens.
    """
    parents = net.get_parents(var)
    descendants = get_descendants(net, var)
    new_givens = {}

    for g in givens:
        if g in parents:
            new_givens[g] = givens[g]
        if g in descendants:
            return givens

    if set(parents) != set(new_givens.keys()):
        return givens
    return new_givens
    
def probability_lookup(net, hypothesis, givens=None):
    "Looks up a probability in the Bayes net, or raises LookupError"
    try:
        if givens:
            givens = simplify_givens(net, list(hypothesis.keys())[0], givens)
        return net.get_probability(hypothesis, givens)
    except(Exception):
        raise LookupError

def probability_joint(net, hypothesis):
    "Uses the chain rule to compute a joint probability"
    prob = 1
    keys = list(hypothesis.keys())
    hypo_sort = reversed(net.topological_sort())
    for var in hypo_sort:
        if var in keys:
            keys.remove(var)
            descendants = get_descendants(net, var)
            given = {}
            for var2 in keys:
                if var2 != var:
                    given[var2] = hypothesis[var2]
            prob *= probability_lookup(net, {var:hypothesis[var]}, given)
    return prob
    
def probability_marginal(net, hypothesis):
    "Computes a marginal probability as a sum of joint probabilities"
    variables = net.get_variables()
    combos = net.combinations(variables, hypothesis)
    prob = 0
    for combo in combos:
        prob+=probability_joint(net, combo)

    return prob

def probability_conditional(net, hypothesis, givens=None):
    "Computes a conditional probability as a ratio of marginal probabilities"

    if givens:
        var = list(hypothesis.keys())[0]
        if var in givens:
            if hypothesis[var] == givens[var]:
                return 1
            else:
                return 0
        return probability_marginal(net,{**hypothesis, **givens})/probability_marginal(net,givens)
    else:
        return probability_marginal(net,hypothesis)
    
def probability(net, hypothesis, givens=None):
    "Calls previous functions to compute any probability"
    if givens:
        return probability_conditional(net,hypothesis,givens)
    return probability_marginal(net,hypothesis)


#### Part 3: Counting Parameters ###############################################

def number_of_parameters(net):
    """
    Computes the minimum number of parameters required for the Bayes net.
    """
    sum = 0
    vars = net.get_variables()
    for var in vars:
        parents_domains = [len(net.get_domain(parent)) for parent in net.get_parents(var)]
        parent_prod = product(parents_domains)
        sum += (len(net.get_domain(var)) - 1) * parent_prod

    return sum



#### Part 4: Independence ######################################################

def is_independent(net, var1, var2, givens=None):
    """
    Return True if var1, var2 are conditionally independent given givens,
    otherwise False. Uses numerical independence.
    """
    combos = net.combinations([var1,var2])
    for combo in combos:
        givens_new = 0
        if givens:
            givens_new = dict(givens)
            givens_new[var2] = combo[var2]
        else:
            givens_new = {var2:combo[var2]}
        if not approx_equal(probability(net, {var1:combo[var1]}, givens), probability(net, {var1:combo[var1]}, givens_new)):
            return False
    return True
    
def is_structurally_independent(net, var1, var2, givens=None):
    """
    Return True if var1, var2 are conditionally independent given givens,
    based on the structure of the Bayes net, otherwise False.
    Uses structural independence only (not numerical independence).
    """

    givens_new = 0
    var_list = set([var1,var2])
    if givens:
        var_list.update(givens.keys())
    for var in set(var_list):
        var_list.update(get_ancestors(net,var))

    subnet = net.subnet(var_list)
    parents = []
    for var in subnet.get_variables():
        parents.append(subnet.get_parents(var))
    for parent_list in parents:
        for par1 in parent_list:
            for par2 in parent_list:
                if par1 != par2:
                    subnet = subnet.link(par1,par2)
    subnet = subnet.make_bidirectional()

    if givens:
        for give in givens:
            subnet = subnet.remove_variable(give)
    if var1 not in subnet.get_variables() or var2 not in subnet.get_variables():
        return True
    return not subnet.find_path(var1, var2)





#### SURVEY ####################################################################

NAME = "Katya Bezugla"
COLLABORATORS = "Julie"
HOW_MANY_HOURS_THIS_LAB_TOOK = 5
WHAT_I_FOUND_INTERESTING = ""
WHAT_I_FOUND_BORING = "None"
SUGGESTIONS = None
