"""
Author:
            Name                Email
        Israel Avihail      bilbisli@gmail.com
"""


import copy

from project_util.convert_to_list import convert_to_list


class DictionaryTree:
    def __init__(self, other=None):
        if other is None:
            other = {}
        self.root = copy.deepcopy(other)

    def add_node(self, path, values=None):
        """
        This method adds a node to the tree and the path (leading nodes) to it if the path doesn't already exists
        :param path: the path to add to the tree
        :type path: Sequence
        :param values: used for adding multiple (more than one) nodes to the end of the specified path (fork)
        :type values: Sequence
        :return: nothing
        :rtype: None
        """
        path = convert_to_list(path)
        values = convert_to_list(values)
        current = self.root
        for p in path[:-1]:
            if p not in current:
                current[p] = {}
            current = current[p]

        current[path[-1]] = {value: {} for value in values} if values else {}

    def get_value_by_path(self, path):
        """
        This method returns the name/s of the node/s (children) under a target (or itself if it is a leaf) node given by
        path to that node together with indication if all nodes under the target node (or itself) are a leaves
        :param path: the path to the target node to retrieve the node/s (children) under the target node
        :type path: Sequence
        :return: if a leaf is reached - returns that leaf with indication (last parameter) that it is a leaf
        otherwise returns the name/s of the node/s reached by the given path with indication if all of them are leaves
        :rtype: (*str, bool)
        """
        current = self.root
        for p in path:
            # condition for flexibility - paths which do not correspond precisely to the tree will still bring a result
            # if some route is valid (ignoring invalid nodes in the path given)
            if p not in current:
                continue
            # condition checks if a leaf is reached (empty dictionary)
            if not current[p]:
                return p, True
            current = current[p]
        # print('current', current)
        return *list(current.keys()), True if not any(current.values()) else False

    def __str__(self):
        return self.root.__str__()


# dt = DictionaryTree()
# dt.add_node(['weather'])
# dt.add_node(['weather', 'sunny', 'parents'], ['no', 'yes'])
# dt.add_node(['weather', 'sunny', 'parents', 'yes', 'cinema'], ['kobi', 'rafi'])
# dt.add_node(['weather', 'sunny', 'parents', 'no', 'yes'])
# dt.add_node(['weather', 'sunny', 'parents', 'no', 'cinema'])
#

# ------
# dt = DictionaryTree()
# dt.add_node(['weather'], ['sunny', 'windy', 'rainy'])
#
# dt.add_node(['weather', 'sunny', 'parents'], ['no', 'yes'])
# dt.add_node(['weather', 'sunny', 'parents', 'no', 'tennis'])
# dt.add_node(['weather', 'sunny', 'parents', 'yes', 'cinema'])
#
# dt.add_node(['weather', 'windy', 'parents'], ['no', 'yes'])
# dt.add_node(['weather', 'windy', 'parents', 'no', 'money'], ['poor', 'rich'])
# dt.add_node(['weather', 'windy', 'parents', 'no', 'money', 'poor', 'cinema'])
# dt.add_node(['weather', 'windy', 'parents', 'no', 'money', 'rich', 'shopping'])
# dt.add_node(['weather', 'windy', 'parents', 'yes', 'cinema'])
# dt.add_node(['weather', 'rainy', 'money', 'rich', 'stay in'])
# dt.add_node(['weather', 'rainy', 'money', 'poor', 'cinema'])

# print(dt)
# print(dt.get_value_by_path(['lol', 'rainy', 'weather', 'rainy', 'money', 'cola', 'cute', 'rich', 'rich']))
# print(dt.get_value_by_path(['lol', 'rainy', 'weather', 'rainy', 'money', 'cola', 'cute', 'rich', 'cinema']))
# print(dt.get_value_by_path(['lol', 'rainy', 'weather', 'rainy', 'money', 'cola', 'cute']))
# print(dt.get_value_by_path(['lol', 'rainy', 'weather']))
# -----


# d['weather']['sunny']['parents']['no']
# d = {'weather': {'sunny': 'sunny', 'windy': 'windy', 'rainy': 'rainy'}}
# d1 = {'weather': {'sunny': {'parents': {'no': 'no', 'yes': 'yes'}}, 'windy': 'windy', 'rainy': 'rainy'}}
# d2 = {'weather': {'sunny': {'parents': {'no': 'cinema', 'yes': 'tennis'}}, 'windy': 'windy', 'rainy': 'rainy'}}
# d3 = {'weather': {'sunny': {'parents': {'no': 'cinema', 'yes': 'tennis'}}, 'windy': {'parents': {'yes': 'yes', 'no': 'no'}}, 'rainy': 'rainy'}}
# d4 = {'weather': {'sunny': {'parents': {'no': 'cinema', 'yes': 'tennis'}}, 'windy': {'parents': {'yes': 'cinema', 'no': {'money': {'poor': 'poor', 'rich': 'rich'}}}}, 'rainy': 'rainy'}}
# d5 = {'weather': {'sunny': {'parents': {'no': 'cinema', 'yes': 'tennis'}}, 'windy': {'parents': {'yes': 'cinema', 'no': {'money': {'poor': 'cinema', 'rich': 'shopping'}}}}, 'rainy': 'rainy'}}

# print(dt)
# d5 = {('root', 'weather'): {('sunny', 'parents'): {'no': 'cinema', 'yes': 'tennis'}}, 'windy': {'parents': {'yes': 'cinema', 'no': {'money': {'poor': 'cinema', 'rich': 'shopping'}}}}, 'rainy': 'rainy'}}