"""
Author:
            Name                Email
        Israel Avihail      bilbisli@gmail.com
"""


from project_util.convert_to_list import convert_to_list
from project_util.entropy import entropy


def information_gain(independent_column, dependant_column):
    """
    This method calculates information gain based on entropy according to given independent column (usually classifier)
    formula: entropy(independent_column) - sum(entropies_of_values(dependent_column))
    :param independent_column: the given independent column to be used for calculating by calculating it's entropy
    :param dependant_column: the dependent column to calculate it's values entropies
    :return: the information gain of the column
    :rtype: float
    """
    independent_column = convert_to_list(independent_column)
    dependant_column = convert_to_list(dependant_column)

    entropies = entropy(independent_column, dependant_column)
    gain = entropy(independent_column)

    for k in entropies:
        # print(dependant_column.count(k), len(dependant_column), entropies[k])
        gain -= (dependant_column.count(k) / len(dependant_column)) * entropies[k]
    return gain


# lst = pd.Series({'weather':  ['sunny']*2 + ['windy'] + ['rainy']*3 + ['windy']*3 + ['sunny']})
# classifier = pd.Series({'decision': ['cinema', 'tennis'] + ['cinema']*2 + ['stay in'] + ['cinema']*2 + ['shopping', 'cinema', 'tennis']})
# print(information_gain(classifier, lst))
