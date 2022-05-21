"""
Author:
            Name                Email
        Israel Avihail      bilbisli@gmail.com
"""

import copy

import pandas as pd


class EqualWidthDiscretization:
    """
    This Class represents equal width data discretization
    """
    def __init__(self, lst, bins=3, labels=None):
        """
        Initializing method
        :param lst: array-like sequence of continuous numerical data to be divided into bins (discretisized)
        :type lst: Sequence
        :param bins: the number of bins to divide the data into
        :type bins: int
        :param labels: the labels (categories) to which the data will be divided into
        :type labels: Sequence
        """
        self.bin_number = bins
        self.partitioned_list = []
        self.ranges = []
        self.labels = labels if labels else range(bins)
        self.lst = lst

        if bins > len(self.labels):
            raise ValueError('bins number must be lower or equal to labels count')

        max_val = max(lst)
        bin_width = (max_val - min(lst)) / bins
        current_range = max_val
        # builds the ranges of the bins
        for bin_ in range(bins - 1):
            bottom = current_range - bin_width
            self.ranges.append(pd.Interval(bottom, current_range))
            current_range -= bin_width
        self.ranges.append(pd.Interval(min(lst) - 0.000000000001, current_range))
        self.ranges = list(reversed(self.ranges))
        # puts the values of the given list into bins
        temp_lst = copy.deepcopy(lst)
        lst_set = sorted(set(temp_lst))
        for i, bin_ in enumerate(self.ranges):
            self.partitioned_list.append([])
            placed_indexes = []
            for index, v in enumerate(lst_set):
                if v in bin_:
                    placed_indexes.append(index)
                    self.partitioned_list[i] += [v] * temp_lst.count(v)
            if placed_indexes:
                del lst_set[placed_indexes[0]:placed_indexes[-1]]
        # print(self.ranges)

    def get_bins(self):
        return self.partitioned_list

    def get_ranges(self):
        return self.ranges

    def get_with_intervals(self):
        """
        This method matches each value with it's range (Interval) and returns the result
        :return: a list of ranges where each value is a range corresponding to the element in the original list
        :rtype: list
        """
        lst = []
        intervals = pd.IntervalIndex(self.ranges)
        for value in self.lst:
            lst.append(intervals[intervals.get_loc(value)])
        return lst

    def get_with_labels(self):
        """
        This method matches each value with it's label and returns the result
        :return: a list of labels where each value is a label corresponding to the element in the original list
        :rtype: list
        """
        lst = []
        temp_list = copy.deepcopy(self.partitioned_list)
        for value in self.lst:
            bin_found = []
            for i, bin_ in enumerate(temp_list):
                if value in bin_:
                    lst.append(self.labels[i])
                    bin_found = bin_
                    break
            bin_found.remove(value)
        return lst


# A3 = [4, 12, 16, 0, 18, 23, 26, 16, 28, 30]
# print(EqualWidthDiscretization(A3, 3).get_ranges())
# print(EqualWidthDiscretization(A3, 3).get_bins())
# print(EqualWidthDiscretization(A3, 3).get_with_intervals())
# print(EqualWidthDiscretization(A3, 3).get_with_labels())
# print(EqualWidthDiscretization(A3, 3, labels=["low", "mid", "high"]).get_with_labels())
