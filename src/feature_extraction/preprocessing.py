from numpy.random import normal


class DemoExtractor:
    def process(self, data):
        return data + normal(0, 0.2, 1)
