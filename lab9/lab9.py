# MIT 6.034 Lab 9: Boosting (Adaboost)
# Written by 6.034 staff

from math import log as ln
from utils import *


#### Part 1: Helper functions ##################################################

def initialize_weights(training_points):
    """Assigns every training point a weight equal to 1/N, where N is the number
    of training points.  Returns a dictionary mapping points to weights."""
    points = {}
    for point in training_points:
        points[point]=make_fraction(1,len(training_points))
    return points

def calculate_error_rates(point_to_weight, classifier_to_misclassified):
    """Given a dictionary mapping training points to their weights, and another
    dictionary mapping classifiers to the training points they misclassify,
    returns a dictionary mapping classifiers to their error rates."""
    error = {}
    for clas, misclas in classifier_to_misclassified.items():
        error[clas] = sum([point_to_weight[point] for point in misclas])
    return error


def pick_best_classifier(classifier_to_error_rate, use_smallest_error=True):
    """Given a dictionary mapping classifiers to their error rates, returns the
    best* classifier, or raises NoGoodClassifiersError if best* classifier has
    error rate 1/2.  best* means 'smallest error rate' if use_smallest_error
    is True, otherwise 'error rate furthest from 1/2'."""
    print('in pick_best')
    if classifier_to_error_rate == {}:
        raise NoGoodClassifiersError
    #print('func: ',classifier_to_error_rate )
    if use_smallest_error:
        items = list(classifier_to_error_rate.items())
        print(items)
        items.sort(key=lambda x: x[0])
        e = min(items, key=lambda x: x[1])[0]
        print('class: ',e,', error: ',classifier_to_error_rate[e])
        #e = min(classifier_to_error_rate,key=classifier_to_error_rate.get)
    else:
        adjusted = {}
        for clas,e in classifier_to_error_rate.items():
            print('class: ',clas,', e: ',e)
            print('distance: ',abs(make_fraction(1,2)-e))
            adjusted[clas] = abs(make_fraction(1,2)-e)
        items = list(adjusted.items())
        items.sort(key=lambda x: x[0])
        e = max(items,key=lambda x: x[1])[0]
        

    if classifier_to_error_rate[e] != make_fraction(1,2):
        return e
    raise NoGoodClassifiersError
    

def calculate_voting_power(error_rate):
    """Given a classifier's error rate (a number), returns the voting power
    (aka alpha, or coefficient) for that classifier."""
    if error_rate == 0:
        return INF
    elif error_rate == 1:
        return -INF
    return make_fraction(1,2) * ln(make_fraction((1-error_rate),error_rate))

def get_overall_misclassifications(H, training_points, classifier_to_misclassified):
    """Given an overall classifier H, a list of all training points, and a
    dictionary mapping classifiers to the training points they misclassify,
    returns a set containing the training points that H misclassifies.
    H is represented as a list of (classifier, voting_power) tuples."""
    misclas = []
    for point in training_points:
        total = 0
        for clas,vote in H:
            if point in classifier_to_misclassified[clas]:
                total+=-1*vote
            else:
                total+=1*vote
        if total <= 0:
            misclas.append(point)
    return set(misclas)


def is_good_enough(H, training_points, classifier_to_misclassified, mistake_tolerance=0):
    """Given an overall classifier H, a list of all training points, a
    dictionary mapping classifiers to the training points they misclassify, and
    a mistake tolerance (the maximum number of allowed misclassifications),
    returns False if H misclassifies more points than the tolerance allows,
    otherwise True.  H is represented as a list of (classifier, voting_power)
    tuples."""
    return len(get_overall_misclassifications(H, training_points,classifier_to_misclassified)) <= mistake_tolerance

def update_weights(point_to_weight, misclassified_points, error_rate):
    """Given a dictionary mapping training points to their old weights, a list
    of training points misclassified by the current weak classifier, and the
    error rate of the current weak classifier, returns a dictionary mapping
    training points to their new weights.  This function is allowed (but not
    required) to modify the input dictionary point_to_weight."""
    new_weight = {}
    for point,old_weight in point_to_weight.items():
        if point in misclassified_points:
            new_weight[point] = make_fraction(1,2) * make_fraction(1,error_rate) * old_weight
        else:
            new_weight[point] = make_fraction(1,2) * make_fraction(1,(1-error_rate)) * old_weight
    return new_weight


#### Part 2: Adaboost ##########################################################

def adaboost(training_points, classifier_to_misclassified,
             use_smallest_error=True, mistake_tolerance=0, max_rounds=INF):
    """Performs the Adaboost algorithm for up to max_rounds rounds.
    Returns the resulting overall classifier H, represented as a list of
    (classifier, voting_power) tuples."""
    print('max_rounds',max_rounds)
    print('use_smallest_error',use_smallest_error)
    count = 0
    H = []

    weights = initialize_weights(training_points)
    #print("class_to_misclass: ", classifier_to_misclassified)
    while count < max_rounds:
        try:
            error_rates = calculate_error_rates(weights, classifier_to_misclassified)
            best_class = pick_best_classifier(error_rates,use_smallest_error)
            best_vote = calculate_voting_power(error_rates[best_class])
            h=(best_class,best_vote)
            H.append(h)
            weights = update_weights(weights,classifier_to_misclassified[best_class],error_rates[best_class])
            #del classifier_to_misclassified[best_class]
            count+=1
            if is_good_enough(H,training_points,classifier_to_misclassified):
                return H
            #print('H',H)
        except NoGoodClassifiersError:
            #print("error met")
            return H
    #print("no error")
    return H

#### SURVEY ####################################################################

NAME = "Katya Bezugla"
COLLABORATORS = "Tara"
HOW_MANY_HOURS_THIS_LAB_TOOK = 6
WHAT_I_FOUND_INTERESTING = ""
WHAT_I_FOUND_BORING = ""
SUGGESTIONS = ""
