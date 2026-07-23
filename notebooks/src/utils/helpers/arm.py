import numpy as np

from collections import defaultdict
from itertools import chain, combinations



def generate_itemsets(iterable, min_len=None, max_len=None):
    s = list(iterable)
    if min_len is None:
        min_len = 0
    if max_len is None:
        max_len = len(s)
    return chain.from_iterable(combinations(sorted(s), r) for r in range(min_len, max_len+1))


def merge_itemsets(a, b):
    itemset = set(a).union(set(b))
    itemset = tuple(sorted(itemset))
    return itemset


def unique_items(db):
    unique_items = set()
    
    for t in db:
        unique_items.update(set(t))
    
    return unique_items


def support_count(db, itemset):
    
    # Set the initial support count to 0
    support_count = 0
    
    # Check for each transaction if it contains the itemset
    # If so, increment support count
    for t in db:
        if set(itemset).issubset(set(t)):
            support_count += 1
            
    # Return support count
    return support_count


def support_itemset(db, itemset):
    
    if len(db) == 0:
        return 0.0
    
    # Return support count
    return support_count(db, itemset) / len(db)


def support(db, rule):
    # Split association rule into itemsets X and Y (reflecting X=>Y)
    X, Y = rule

    # Calculate X*union*Y
    itemset = tuple(sorted(set(X).union(set(Y))))

    # Return support of itemset
    return support_itemset(db, itemset)


def confidence(db, rule):
    # Split association rule into itemsets X and Y (reflecting X=>Y)
    X, Y = rule
    
    # Calculate the support count for X
    support_X = support_itemset(db, X)
    
    # If the support count of X is 0, return 0 to avoid division by zero
    if support_X == 0:
        return 0.0
    
    # Calculate X*union*Y
    itemset = tuple(sorted(set(X).union(set(Y))))
    
    # Caluculate and return the confidence
    return support_itemset(db, itemset) / support_X


def lift(db, rule):
    # Split association rule into itemsets X and Y (reflecting X=>Y)
    X, Y = rule
    
    # Calculate the support for X and Y
    support_X = support_itemset(db, X)
    support_Y = support_itemset(db, Y)
    
    # If the support count of X is 0, return 0 to avoid division by zero
    if support_X == 0 or support_Y == 0:
        return 0.0
    
    # Calculate X*union*Y
    itemset = tuple(sorted(set(X).union(set(Y))))
    
    # Caluculate and return the confidence
    return support_itemset(db, itemset) / (support_X * support_Y)


def generate_rules(itemset):
    rules = []
    
    for X in generate_itemsets(itemset, min_len=1, max_len=len(itemset)-1):
        Y = tuple(sorted(set(itemset).difference(set(X))))
        rules.append((X, Y))
    
    return rules