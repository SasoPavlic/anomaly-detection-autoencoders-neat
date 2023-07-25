from configparser import ConfigParser

from sklearn.metrics import auc
from sklearn.metrics import roc_curve, mean_squared_error

from neat.config import Config


class AnomalyDetectionConfig(Config):
    def __init__(self, AutoencoderGenome, DefaultReproduction, DefaultSpeciesSet, DefaultStagnation, filename):
        super().__init__(AutoencoderGenome, DefaultReproduction, DefaultSpeciesSet, DefaultStagnation, filename)

        config = ConfigParser()
        self.path = filename
        config.read(self.path)
        self.generations = int(config.get('NEAT', 'generations'))
        self.anomaly_label = str(config.get('AnomalyDetection', 'anomaly_label'))
        self.curriculum_levels = str(config.get('AnomalyDetection', 'curriculum_levels'))
        self.data_percentage = float(config.get('AnomalyDetection', 'data_percentage'))
        self.test_size = float(config.get('AnomalyDetection', 'test_size'))


class AnomalyDetection(object):

    def __init__(self, X_train, X_test, y_train, y_test, valid_label, anomaly_label, all_generations,
                 curriculum_levels):
        # TODO make it more efficient
        # Convert pandas dataframe to numpy array [:-1] to remove label column Level
        self.train_X_array = X_train.to_numpy()
        self.train_Y_array = y_train.to_numpy()
        self.test_X_array = X_test.to_numpy()
        self.test_Y_array = y_test

        self.valid_label = valid_label
        self.anomaly_label = anomaly_label
        self.all_generations = all_generations
        self.curriculum_levels = curriculum_levels
        self.acc_list = []

        self.metrics = []
        self.FPR_array = None
        self.TPR_array = None
        self.roc_auc = None
        self.AUC = None

    def calculate_mse(self, encoder, decoder, generation):
        """Calculate mean squared error between original and reconstructed data
        """
        decoded_instances = []
        scores = []
        targets = []
        data = None

        for x, y in zip(self.train_X_array, self.train_Y_array):

            gen_percentage = generation / self.all_generations * 100

            if self.curriculum_levels == 'zero':
                data = x[:-1]


            # TODO make generation percentage parametric

            elif self.curriculum_levels == 'two':
                if gen_percentage <= 60:
                    if x[-1] == 'Easy':
                        data = x[:-1]
                    else:
                        continue
                elif gen_percentage > 60:
                    if x[-1] == 'Hard':
                        data = x[:-1]
                    else:
                        continue

            elif self.curriculum_levels == 'three':
                if gen_percentage <= 50:
                    if x[-1] == 'Easy':
                        data = x[:-1]
                    else:
                        continue

                elif 50 < gen_percentage <= 80:
                    if x[-1] == 'Medium':
                        data = x[:-1]
                    else:
                        continue

                elif gen_percentage > 80:
                    if x[-1] == 'Hard':
                        data = x[:-1]
                    else:
                        continue

            bottle_neck = encoder.activate(data)
            decoded = decoder.activate(bottle_neck)

            decoded_instances.append(decoded)
            targets.append(y)
            mse = mean_squared_error(data, decoded)
            # rmse = math.sqrt(mean_squared_error(data, decoded))
            scores.append(mse)

        # Return mse_list mean value
        return decoded_instances, scores, targets

    def calculate_final_mse(self, encoder, decoder):
        decoded_instances = []
        scores = []
        targets = []
        data = None
        for x, y in zip(self.test_X_array, self.test_Y_array):
            data = x[:-1]

            bottle_neck = encoder.activate(data)
            decoded = decoder.activate(bottle_neck)

            decoded_instances.append(decoded)
            targets.append(y)
            mse = mean_squared_error(data, decoded)
            # rmse = math.sqrt(mean_squared_error(data, decoded))
            scores.append(mse)

        # Return mse_list mean value
        return decoded_instances, scores, targets

    def calculate_roc_auc_curve(self, encoder, decoder):
        # https://stackoverflow.com/questions/58894137/roc-auc-score-for-autoencoder-and-isolationforest

        decoded_instances, scores, targets = self.calculate_final_mse(encoder, decoder)

        try:
            self.FPR_array = dict()
            self.TPR_array = dict()
            thresholds = dict()
            self.roc_auc = dict()
            for i in range(2):
                self.FPR_array[i], self.TPR_array[i], thresholds[i] = roc_curve(targets, scores)
                self.roc_auc[i] = auc(self.FPR_array[i], self.TPR_array[i])

            self.AUC = round(self.roc_auc[0], 3)

        except Exception as e:
            print(e)
            self.AUC = 0.0
