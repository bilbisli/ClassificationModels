"""
Author:
            Name                Email
        Israel Avihail      bilbisli@gmail.com
"""

class Container:
    def __init__(self, *extras, **kwextras):
        self.extras = extras
        self.kwextras = kwextras

    def get_extras(self):
        return self.extras

    def get_kwextras(self):
        return self.kwextras

    def get_extra_by_name(self, name):
        return self.kwextras[name] if self.kwextras and name in self.kwextras else None

    def get_extra_by_index(self, index):
        return self.extras[index] if self.extras and len(self.extras) > index else None