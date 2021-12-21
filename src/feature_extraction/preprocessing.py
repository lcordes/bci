from numpy.random import normal
from random import randint


class DemoExtractor:
    def process(self, data):
        return randint(0, 2) + normal(0, 0.2, 1)
