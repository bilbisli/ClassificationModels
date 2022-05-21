"""
Author:
            Name                Email
        Israel Avihail      bilbisli@gmail.com
"""

import copy

import pandas as pd


class EqualDepthDiscretization:
    """
    This Class represents equal depth (frequency) data discretization
    Examples:
        A = [9, 15, 21, 24, 25, 4, 26, 21, 34, 8]
        1. EqualDepthDiscretization(A, 3).get_bins()   ->  [[4, 8, 9, 15], [21, 21, 24], [25, 26, 34]]

        2. EqualDepthDiscretization(A, 3, labels=["low", "mid", "high"]).get_with_labels() ->
        ['low', 'low', 'mid', 'mid', 'high', 'low', 'high', 'mid', 'high', 'low']

        3. EqualDepthDiscretization(A, 2, labels=["low", "mid", "high"]).get_with_labels() ->
        ['low', 'low', 'low', 'mid', 'mid', 'low', 'mid', 'mid', 'mid', 'low']

        4. EqualDepthDiscretization(A3, 3).get_with_intervals() ->
        [Interval(3.999999999999, 18.0, closed='right'),
        Interval(3.999999999999, 18.0, closed='right'),
        Interval(18.0, 24.5, closed='right'),
        Interval(18.0, 24.5, closed='right'),
        Interval(24.5, 34, closed='right'),
        Interval(3.999999999999, 18.0, closed='right'),
        Interval(24.5, 34, closed='right'),
        Interval(18.0, 24.5, closed='right'),
        Interval(24.5, 34, closed='right'),
        Interval(3.999999999999, 18.0, closed='right')]
    """
    def __init__(self, lst, bins=3, is_sorted=False, labels=None):
        """
        Initializing method
        :param lst: array-like sequence of continuous numerical data to be divided into bins (discretisized)
        :type lst: Sequence
        :param bins: the number of bins to divide the data into
        :type bins: int
        :param is_sorted: indication if the sequence is sorted or not (to avoid resorting)
        :type is_sorted: bool
        :param labels: the labels (categories) to which the data will be divided into
        :type labels: Sequence
        """
        self.bin_number = bins
        self.partitioned_list = []
        self.ranges = []
        self.lst = lst

        if len(lst) <= bins:
            bins = len(lst)

        self.labels = labels if labels else range(1, bins + 1)
        if bins > len(self.labels):
            raise ValueError('bins number must be lower or equal to labels count')
        self.label_indexes = {k: v for k, v in zip(self.labels, range(len(self.labels)))}

        bin_size = len(lst) // bins

        for _ in range(bins):
            self.partitioned_list.append([])
        sorted_list = lst if is_sorted else sorted(lst, reverse=False)
        i, j = 0, 0
        m = len(lst) % bins
        # puts the values of the given list into bins
        for v in sorted_list:
            if m > 0 and len(self.partitioned_list[i]) == bin_size:
                self.partitioned_list[i].append(v)
                m -= 1
                continue
            if (len(self.partitioned_list[i]) >= bin_size) and (i + 1 < bins):
                i += 1
            self.partitioned_list[i].append(v)
        # builds the ranges of the bins
        bottom = min(lst) - 0.000000000001
        for bin_index in range(bins - 1):
            top = (max(self.partitioned_list[bin_index]) + min(self.partitioned_list[bin_index + 1])) / 2
            self.ranges.append(pd.Interval(bottom, top))
            bottom = top
        self.ranges.append(pd.Interval(bottom, max(lst)))
        # turns redundant ranges (which both left and right bounds are equal) to be their left neighbour
        self.ranges = [self.ranges[i] if range_.right != range_.left else self.ranges[i-1]
                       for i, range_ in enumerate(self.ranges)]

    def get_ranges(self):
        return self.ranges

    def get_bins(self):
        return self.partitioned_list

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

    def get_with_intervals(self):
        """
        This method is the same (and uses) get_with_labels method only the values are represented as their corresponding
        ranges (Intervals)
        :return: a list of ranges where each value is a range corresponding to the element in the original list
        :rtype: list
        """
        return [self.ranges[self.label_indexes[val]] for val in self.get_with_labels()]


# A3 = [9, 15, 21, 24, 25, 4, 26, 21, 34, 8]
# # A3 = [1, 1, 1, 1]
# disc = EqualDepthDiscretization(A3, 3)
#
# print(disc.get_ranges())
# print(1 in disc.get_ranges()[1])
# print(disc.get_bins())
# print(disc.get_with_intervals())
# print(EqualDepthDiscretization(A3, 3, labels=["low", "mid", "high"]).get_with_labels())
# print(EqualDepthDiscretization(A3, 2, labels=["low", "mid", "high"]).get_with_labels())
# # print(EqualDepthDiscretization(A3, 4, labels=["low", "mid", "high"]).get_with_labels())
# print(EqualDepthDiscretization(A3, 3).get_with_intervals())
