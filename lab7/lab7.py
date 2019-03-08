# MIT 6.034 Lab 7: Support Vector Machines
# Written by 6.034 staff

from svm_data import *
from functools import reduce


#### Part 1: Vector Math #######################################################

def dot_product(u, v):
    """Computes the dot product of two vectors u and v, each represented 
    as a tuple or list of coordinates. Assume the two vectors are the
    same length."""
    return sum([u[i]*v[i] for i in range(len(u))])

def norm(v):
    """Computes the norm (length) of a vector v, represented 
    as a tuple or list of coords."""
    return (sum([v[i]*v[i] for i in range(len(v))]))**(1/2)


#### Part 2: Using the SVM Boundary Equations ##################################

def positiveness(svm, point):
    """Computes the expression (w dot x + b) for the given Point x."""
    return dot_product(svm.w, point) + svm.b

def classify(svm, point):
    """Uses the given SVM to classify a Point. Assume that the point's true
    classification is unknown.
    Returns +1 or -1, or 0 if point is on boundary."""
    num = positiveness(svm, point)
    return 0 if num == 0 else num/abs(num)

def margin_width(svm):
    """Calculate margin width based on the current boundary."""
    return 2/norm(svm.w)

def check_gutter_constraint(svm):
    """Returns the set of training points that violate one or both conditions:
        * gutter constraint (positiveness == classification, for support vectors)
        * training points must not be between the gutters
    Assumes that the SVM has support vectors assigned."""
    points = []
    for point in svm.training_points:
        if (positiveness(svm, point) != point.classification and point in svm.support_vectors) or abs(positiveness(svm, point)) < 1:
            points.append(point)
    return set(points)


#### Part 3: Supportiveness ####################################################

def check_alpha_signs(svm):
    """Returns the set of training points that violate either condition:
        * all non-support-vector training points have alpha = 0
        * all support vectors have alpha > 0
    Assumes that the SVM has support vectors assigned, and that all training
    points have alpha values assigned."""
    points = []
    for point in svm.training_points:
        if (point in svm.support_vectors and point.alpha <= 0) or (point not in svm.support_vectors and point.alpha != 0):
            points.append(point)
    return set(points)

def check_alpha_equations(svm):
    """Returns True if both Lagrange-multiplier equations are satisfied,
    otherwise False. Assumes that the SVM has support vectors assigned, and
    that all training points have alpha values assigned."""

    eq4 = 0
    eq5 = None
    for point in svm.training_points:
        eq4 += point.classification * point.alpha
        if eq5 is None:
            eq5 = scalar_mult(point.classification * point.alpha, point.coords)
        else:
            eq5 = vector_add(eq5,scalar_mult(point.classification * point.alpha, point.coords))
    if eq4 == 0 and eq5 == svm.w:
        return True
    return False


#### Part 4: Evaluating Accuracy ###############################################

def misclassified_training_points(svm):
    """Returns the set of training points that are classified incorrectly
    using the current decision boundary."""
    points = []
    for point in svm.training_points:
        if classify(svm, point) != point.classification:
            points.append(point)
    return set(points)


#### Part 5: Training an SVM ###################################################

def update_svm_from_alphas(svm):
    """Given an SVM with training data and alpha values, use alpha values to
    update the SVM's support vectors, w, and b. Return the updated SVM."""
    svm.support_vectors = []
    w = [0,0]
    for point in svm.training_points:
        if point.alpha != 0:
            svm.support_vectors.append(point)
            w = vector_add(w,scalar_mult(point.classification * point.alpha, point.coords))
    neg = []
    pos = []
    for point in svm.support_vectors:
        if point.classification < 0:
            neg.append(point.classification - dot_product(w,point))
        elif point.classification > 0:
            pos.append(point.classification - dot_product(w,point))
    b = (min(neg) + max(pos))/2

    return svm.set_boundary(w, b)


#### Part 6: Multiple Choice ###################################################

ANSWER_1 = 11
ANSWER_2 = 6
ANSWER_3 = 3
ANSWER_4 = 2

ANSWER_5 = ['A','D']
ANSWER_6 = ['A', 'B', 'D']
ANSWER_7 = ['A', 'B', 'D']
ANSWER_8 = []
ANSWER_9 = ['A', 'B', 'D']
ANSWER_10 = ['A', 'B', 'D']

ANSWER_11 = False
ANSWER_12 = True
ANSWER_13 = False
ANSWER_14 = False
ANSWER_15 = False
ANSWER_16 = True

ANSWER_17 = [1,3, 6, 8]
ANSWER_18 = [1, 2, 4, 5, 6, 7, 8]
ANSWER_19 = [1,2,4,5,6,7,8]

ANSWER_20 = 6


#### SURVEY ####################################################################

NAME = "Katya Bezugla"
COLLABORATORS = "None"
HOW_MANY_HOURS_THIS_LAB_TOOK = 3
WHAT_I_FOUND_INTERESTING = "None"
WHAT_I_FOUND_BORING = "None"
SUGGESTIONS = "None"
