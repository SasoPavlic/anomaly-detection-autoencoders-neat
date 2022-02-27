import numpy as np
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from configparser import ConfigParser
from neat.config import Config


class AnomalyDetectionConfig(Config):

    def __init__(self, AutoencoderGenome, DefaultReproduction, DefaultSpeciesSet, DefaultStagnation, filename):
        super().__init__(AutoencoderGenome, DefaultReproduction, DefaultSpeciesSet, DefaultStagnation, filename)

        config = ConfigParser()
        self.path = filename
        config.read(self.path)
        self.anomaly_label = str(config.get('AnomalyDetection', 'anomaly_label'))


class Metric(object):
    def __init__(self, quantile, threshold, outliers_idx, quantile_instance_labels):
        self.quantile = quantile
        self.threshold = threshold
        self.outliers_idx = outliers_idx
        self.quantile_instance_labels = quantile_instance_labels
        self.anomaly_count = None
        self.valid_count = None
        self.TP = None
        self.FN = None
        self.FP = None
        self.TN = None
        self.TPR = None
        self.FNR = None
        self.TNR = None
        self.FPR = None

    def evaluate_anomalies(self, TP, FN, FP, TN):
        """Compute recall, precision and F1-score
        Returns:
            Recall, Precision, F1-score
        """
        self.accuracy = ((TP + TN) / (TP + TN + FP + FN))
        self.recall = (TP / (TP + FN))
        self.precision = ((TP / (TP + FP)))
        self.F1 = 2 * ((self.precision * self.recall) / (self.precision + self.recall))

    def calculate_confusion_matrix(self, y_test, valid_label, anomaly_label):
        """Compute confusion matrix based on found anomalies in dataset
        """
        self.anomaly_count = sum(x in anomaly_label for x in y_test)
        self.valid_count = sum(x in valid_label for x in y_test)
        self.TP = sum(sum(x == anomaly_label for x in self.quantile_instance_labels))
        self.FN = self.anomaly_count - self.TP
        self.FP = len(self.outliers_idx) - self.TP
        self.TN = self.valid_count - self.FP

        self.TPR = (self.TP / (self.TP + self.FN))
        self.FNR = (self.FN / (self.TP + self.FN))
        self.TNR = (self.TN / (self.TN + self.FP))
        self.FPR = 1 - self.TNR

        self.evaluate_anomalies(self.TP, self.FN, self.FP, self.TN)


class AnomalyDetection(object):

    def __init__(self, x_test, y_test, valid_label, anomaly_label):
        self.x_test = x_test.to_numpy()
        self.y_test = y_test
        self.valid_label = valid_label
        self.anomaly_label = anomaly_label

        self.metrics = []
        self.FPR_array = []
        self.TPR_array = []
        self.AUC = None

    def find(self, encoder, decoder, final_evaluation=False):
        """Compute ROC-AUC values and visualize it in plot
        """
        self.AUC = None
        self.metrics = []
        self.FPR_array = []
        self.TPR_array = []

        decoded_instances = []
        for i, x in enumerate(self.x_test):
            bottle_neck = encoder.activate(x)
            decoded = decoder.activate(bottle_neck)
            decoded_instances.append(decoded)

        errors = []

        # loop over all original images and their corresponding
        # reconstructions
        for (image, recon) in zip(self.x_test, decoded_instances):
            # compute the mean squared error between the ground-truth image
            # and the reconstructed image, then add it to our list of errors
            mse = np.mean((image - recon) ** 2)
            errors.append(mse)

        for quantile in np.linspace(0, 1, 100):
            threshold = np.quantile(errors, quantile)
            outliers_idx = np.where(np.array(errors) >= threshold)[0]
            quantile_instance_labels = np.array(self.y_test)[outliers_idx.astype(int)]

            metric = Metric(quantile, threshold, outliers_idx, quantile_instance_labels)

            metric.calculate_confusion_matrix(self.y_test, self.valid_label, self.anomaly_label)
            self.metrics.append(metric)

            self.FPR_array.append(metric.FPR)
            self.TPR_array.append(metric.TPR)

        self.AUC = abs(round(np.trapz(self.TPR_array, self.FPR_array), 3))


