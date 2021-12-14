class Predictor:
    def __init__(self, model_file):
        pass  # load model here

    def predict(self, data):
        pred = round(float(data))
        if pred < 0 or pred > 2:
            pred = 2
        return pred
