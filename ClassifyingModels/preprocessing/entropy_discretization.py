"""
Author:
            Name                Email
        Israel Avihail      bilbisli@gmail.com
"""

import copy
import operator
import pandas as pd
from itertools import combinations
from typing import Sequence
from scipy.stats import entropy


def generate_cut_indexes(bins, data_size, start=0):
    bins = range(bins) if not isinstance(bins, Sequence) else bins

    for indexes_size in range(1, len(bins)):
        for indexes in combinations(range(start + 1, data_size), indexes_size):
            yield list((start,) + indexes + (data_size,))


def generate_entropies(start_index, end_index, data_as_set, classifiers):
    classifiers_calc = {}
    for classifier_ in classifiers:
        classifiers_calc[classifier_] = 0

    classifiers_size = 0
    percentile = 0
    for i in range(start_index, end_index):
        percentile += data_as_set[i]['percentile']
        for classifier_ in classifiers:
            classifiers_calc[classifier_] += data_as_set[i][classifier_]
            classifiers_size += data_as_set[i][classifier_]

    probability_list = [classifiers_calc[classify] / classifiers_size for classify in
                        classifiers]

    yield percentile * entropy(probability_list, base=2)


def calc_net_entropy(data_as_set, bins, classifiers, start=0, total_data_size=None, memorization=None):
    memorization = {} if memorization is None else memorization
    total_data_size = total_data_size if total_data_size else len(data_as_set)
    for index_list in generate_cut_indexes(bins, total_data_size, start):
        net_entropy = 0
        for current in range(1, len(index_list)):
            edges = index_list[current - 1], index_list[current]
            try:
                net_entropy += memorization[edges]
            except KeyError:
                memorization[edges] = next(generate_entropies(*edges, data_as_set, classifiers))
                net_entropy += memorization[edges]

        yield [index_list, net_entropy]


class EntropyDiscretization:
    def __init__(self, dependant_column, independent_column, bins=2, minimal_gain=0.0):

        self.dependant_column = dependant_column
        self.independent_column = independent_column
        self.data_size = self.dependant_column.size
        self.classifiers = independent_column.value_counts()
        self.classifiers = {k: v for k, v in zip(self.classifiers.index.tolist(), self.classifiers.tolist())}
        self.bins = bins
        self.minimal_gain = minimal_gain
        self.split_list = []
        self.memorization = {}

        united_df = pd.DataFrame()
        united_df['class'] = independent_column
        united_df['column'] = dependant_column

        data_as_set = dependant_column.value_counts()
        # create a set of the unique values with their count
        data_as_set = [{'value': v, 'count': c} for v, c in zip(data_as_set.index.tolist(),
                                                                data_as_set.tolist())]
        data_as_set.sort(key=lambda vc: vc['value'])
        counts = independent_column.value_counts()
        self.general_entropy = entropy(counts, base=2)
        for value in data_as_set:
            value['percentile'] = value['count'] / self.data_size
            for classifier_ in counts.index.tolist():
                value[classifier_] = united_df[(united_df['column'] == value['value']) &
                                               (united_df['class'] == classifier_)]['class'].count()
        self.data_as_set = data_as_set
        self.cut_indexes = [0, len(data_as_set)]

    def split(self, make_split):
        if not make_split or make_split['info_gain'] < self.minimal_gain:
            return
        cut = make_split['cut_index']
        self.cut_indexes.append(cut)
        self.cut_indexes = list(sorted(set(self.cut_indexes)))
        if len(self.cut_indexes) - 1 >= self.bins or len(make_split['data']) - make_split['start'] <= 1:
            return
        first_sub_split = self.get_gain_and_indexes(make_split['data'][:cut])
        second_sub_split = self.get_gain_and_indexes(make_split['data'], start=cut)
        self.split_list += ([first_sub_split] if first_sub_split else []) \
                           + ([second_sub_split] if second_sub_split else [])
        if self.split_list:
            next_split = max(self.split_list, key=lambda split_dict: split_dict['info_gain'])
            self.split_list.remove(next_split)
            self.split(make_split=next_split)

    def get_gain_and_indexes(self, data, start=0):
        if len(self.cut_indexes) - 1 >= self.bins or len(data) - start <= 1:
            return {}
        classifier_counter = {k: 0 for k in self.classifiers}
        total_classifiers = 0
        for classifier in self.classifiers:
            for value in data[start:]:
                classifier_counter[classifier] += value[classifier]
                total_classifiers += value[classifier]
        probability_list = []
        for classifier in self.classifiers:
            probability_list.append(classifier_counter[classifier] / total_classifiers)

        general_entropy = entropy(probability_list, base=2)

        best_net_entropy = list(min(calc_net_entropy(data, 2, self.classifiers, start=start),
                                    key=operator.itemgetter(1)))
        # offset the net entropy calculated to match the data calculated instead of matching the
        # original (complete) data since the percentiles of each values are calculated in calc_net_entropy based on the
        # original data because the data_as_set was built on it
        offset = self.data_size / sum(value['count'] for value in data[start:])
        best_split = {'info_gain': general_entropy - best_net_entropy[1] * offset,
                      'cut_index': best_net_entropy[0][1],
                      'data': data,
                      'start': start,
                      }
        return {} if best_split['info_gain'] < self.minimal_gain else best_split

    def best_cut_quantiles(self):
        data = copy.deepcopy(self.data_as_set)
        if self.bins <= 1:
            return 0.0, 1.0
        first_split = self.get_gain_and_indexes(data)
        self.split(make_split=first_split)

        add_percentiles = 0.0
        quantile_list = [0.0]
        cut_indexes = list(sorted(set(self.cut_indexes)))
        cut_indexes.pop(-1)
        cut_indexes.pop(0)
        prev = 0
        for i in cut_indexes:
            for k in range(prev, i):
                add_percentiles += self.data_as_set[k]['percentile']
            prev = i
            quantile_list.append(add_percentiles)
        quantile_list.append(1.0)

        return quantile_list

    def apply_discretization(self):
        return pd.qcut(self.dependant_column, q=self.best_cut_quantiles())
