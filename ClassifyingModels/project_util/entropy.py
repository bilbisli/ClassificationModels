"""
Author:
            Name                Email
        Israel Avihail      bilbisli@gmail.com
"""


from fractions import Fraction
from math import log2
from project_util.convert_to_list import convert_to_list


def entropy(independent_column, dependant_column=None):
    """
    This function calculates entropy either given by probability\value sequence\dataframe\series for regular entropy
    or entropies of a dependant column that depends on an independent column
    :param independent_column: the independent column to calculate it's entropy or to be used for calculating
     the dependent column entropy if given
    :param dependant_column: the dependent column to calculate it's entropies
    :return: the entropy of a single column or the entropies of all values in the dependent column based on the
    corresponding values within the independent column
    :rtype: float, dict(str: float,)
    """

    independent_column = convert_to_list(independent_column)
    dependant_column = convert_to_list(dependant_column)
    # for the occasion that a dependent column value entropies are to be calculated and not a single columns \ sequence
    # of entropies is given
    if dependant_column and len(dependant_column) == len(independent_column):
        entropies = {}
        pairs = list(zip(dependant_column, independent_column))
        # constructs a probability list for each value in the dependant column and calculates it's entropy based on the
        # independent column
        for value in set(dependant_column):
            probability_list = []
            count = dependant_column.count(value)
            for pair in set(pairs):
                if value == pair[0]:
                    probability_list.append(Fraction(pairs.count(pair), count))
            # recursive call to calculate the entropy
            entropies[value] = entropy(probability_list)
        # returns the dictionary of entropies of each value in the dependant column
        return entropies
    # for the occasion that the given independent column is not a probability list
    if not (all(isinstance(x, (int, float, Fraction)) for x in independent_column)
            and (0.99999999999 <= sum(independent_column) <= 1.00000000001)):
        independent_column = [Fraction(independent_column.count(value), len(independent_column))
                              for value in set(independent_column)]
    return sum(-p * log2(p) for p in independent_column)



