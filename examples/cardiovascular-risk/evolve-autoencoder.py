import math
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.base import OutlierMixin
# from sklearn_pandas import DataFrameMapper
from sklearn.model_selection import train_test_split

import neat
import visualize
from anomalyDetection import AnomalyDetectionConfig, AnomalyDetection


class UnsupervisedMixin:
    def fit(self, X, y=None):
        """Fit the model using X as training data

        Parameters
        ----------
        X : {array-like, sparse matrix, BallTree, KDTree}
            Training data. If array or matrix, shape [n_samples, n_features],
            or [n_samples, n_samples] if metric='precomputed'.
        """
        return self._fit(X)


"""
Constants:
    - MAX_ACCUARCY depends on chosen dataset, try to manually tweak it.
"""
GENERATIONS = 100
anomaly_detection = None
saving_path = f"./logs/{GENERATIONS}_generations"


def eval_genomes(genomes, config, X):
    global current_best_accuracy
    for genome_id, genome in genomes:
        encoder, decoder = neat.nn.FeedForwardNetwork.create_autoencoder(genome, config)

        # anomaly_detection.calculate_roc_auc_curve(encoder, decoder)
        decoded_instances, scores, targets = anomaly_detection.calculate_mse(encoder, decoder)

        genome.fitness = int(np.mean(scores) * math.pow(10, 15))

        # Write genome.fitness to file line by line
        with open('fitness.txt', 'a+') as f:
            f.write(str(genome.fitness) + '\n')
            #print(f"Genome {genome_id} fitness: {genome.fitness}")



class NeatOutlier(OutlierMixin, UnsupervisedMixin):
    def __init__(self, config, generations=100):
        super().__init__()
        self.generations = generations
        self.encoder = None
        self.decoder = None
        self.config = config
        self.stats = None
        self.winner = None

    def fit(self, X, y=None):
        population = neat.Population(self.config)
        self.stats = neat.StatisticsReporter()

        population.add_reporter(self.stats)
        population.add_reporter(neat.StdOutReporter(True))

        self.winner = population.run(eval_genomes, X, self.generations, anomaly_detection, neat, self.config)
        # Save the winner.
        os.makedirs(saving_path, exist_ok=True)
        os.makedirs(saving_path + '/winner-AE', exist_ok=True)

        self.stats.save(saving_path)

        with open(saving_path + '/winner-AE' + '/best_model.pkl', 'wb+') as f:
            pickle.dump(self.winner, f)

        self.encoder, self.decoder = neat.nn.FeedForwardNetwork.create_autoencoder(self.winner, self.config)

        return self


if __name__ == '__main__':
    start = datetime.now().strftime("%H:%M:%S-%d/%m/%Y")
    print(f"Program start... {start}")

    """Prepare dataset for anomaly detection based on configuration file"""
    config = AnomalyDetectionConfig(neat.AutoencoderGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                                    neat.DefaultStagnation, 'evolve-autoencoder.cfg')

    """Select the dataset"""
    with open("../../datasets/CVD_formated.csv") as f:
        dataset = pd.read_csv(f, delimiter=",").sample(frac=1, random_state=0)#.head(333)
        data = dataset.iloc[:, :19]
        target = dataset["Heart_Disease"]


    # with open("../../datasets/fault_detection.csv") as f:
    #     dataset = pd.read_csv(f, delimiter=";").sample(frac=1, random_state=0).head(300)
    #     data = dataset.iloc[:, :60]
    #     target = dataset["Fault_lag"]

    """
        Since we are interested to perform unsupervised* anomaly detection, we will help
        the algorithm by reducing the amount of anomalous data instances. By doing this
        algorithm learn better to encode/decode majority of data (normal instances).
        As a result minority of data (anomalies) will be reconstructed worst and therefore,
        they will be more likely to be mark as a true positives anomalies.

        *Disclaimer: One could argue that this is in fact a semi-supervised anomaly detection technique...
    """

    print(f"Anomaly label is for class: {config.anomaly_label}")

    """Link is explaining the bellow step"""
    # https://towardsdatascience.com/what-and-why-behind-fit-transform-vs-transform-in-scikit-learn-78f915cf96fe
    # mapper = DataFrameMapper([(data.columns, StandardScaler())])
    # scaled_features = mapper.fit_transform(data.copy())
    data = pd.DataFrame(data, index=data.index, columns=data.columns)

    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.33, random_state=42)

    """If supervised anomaly detection is intended, then we can remove anomalies from training"""
    # X_train = X_train[y_train != config.anomaly_label, :]

    """Turning multiple class data labels to binary"""
    y_temp = []
    for yi in y_test:
        if str(yi) == config.anomaly_label:
            y_temp.append(1)
        else:
            y_temp.append(0)

    y_test = y_temp

    # anomaly_detection = AnomalyDetection([0], [1])
    anomaly_detection = AnomalyDetection(X_test, y_test, [0], [1])

    """Adjust number of generations"""
    neat_outlier = NeatOutlier(config, generations=GENERATIONS)

    """Perform neuroevolution on training dataset"""
    NO = neat_outlier.fit(X_train.to_numpy())

    """Plotting results from anomaly detection on graph"""

    anomaly_detection.calculate_roc_auc_curve(neat_outlier.encoder, neat_outlier.decoder)
    print(f"=====================================")
    print(f"Model AUC score: {anomaly_detection.AUC}")

    visualize.plot_roc_curve(anomaly_detection.roc_auc, anomaly_detection.FPR_array, anomaly_detection.TPR_array, True,
                             filename=saving_path + "/roc_curve.svg")
    # visualize.plot_metrics(anomaly_detection.metrics, True, filename=saving_path + "/metrics.svg")

    visualize.draw_net_encoder(config, NO.winner.encoder, False,
                               node_colors={key: 'yellow' for key in NO.winner.encoder.nodes},
                               show_disabled=True, filename=saving_path + "/encoder")

    visualize.draw_net_decoder(config, NO.winner.decoder, False,
                               node_colors={key: 'yellow' for key in NO.winner.decoder.nodes},
                               show_disabled=True, filename=saving_path + "/decoder")

    visualize.plot_stats(NO.stats, ylog=False, view=False, filename=saving_path + "/fitness.svg")
    visualize.plot_species(NO.stats, view=False, filename=saving_path + "/speciation.svg")

    end = datetime.now().strftime("%H:%M:%S-%d/%m/%Y")
    print(f"Program end... {end}")
