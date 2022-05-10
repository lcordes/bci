class ConcentrationExtractor:
    def __init__(self, sr, board_id):
        self.sr = sr
        self.board_id = board_id

    def get_concentration(self, data):
        eeg_channels = BoardShim.get_eeg_channels(int(self.board_id))
        data = np.squeeze(data)
        bands = DataFilter.get_avg_band_powers(data, eeg_channels, self.sr, True)
        feature_vector = np.concatenate((bands[0], bands[1]))

        # calc concentration
        concentration_params = BrainFlowModelParams(
            BrainFlowMetrics.CONCENTRATION.value, BrainFlowClassifiers.KNN.value
        )
        concentration = MLModel(concentration_params)
        concentration.prepare()
        conc = int(concentration.predict(feature_vector) * 100)
        concentration.release()

        return conc
