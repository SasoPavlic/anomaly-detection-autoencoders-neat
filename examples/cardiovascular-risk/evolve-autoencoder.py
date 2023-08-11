import argparse
import os
import pickle
import uuid
from datetime import datetime

import numpy as np
import pandas as pd
# from sklearn_pandas import DataFrameMapper
from sklearn.model_selection import train_test_split

import dataloader as data_loader
import neat
import visualize
from anomalyDetection import AnomalyDetectionConfig, AnomalyDetection

"""
Constants:
    - MAX_ACCUARCY depends on chosen dataset, try to manually tweak it.
"""
RUN_UUID = uuid.uuid4().hex
anomaly_detection = None
global saving_path


def eval_genomes(genomes, config, generation):
    for genome_id, genome in genomes:
        encoder, decoder = neat.nn.FeedForwardNetwork.create_autoencoder(genome, config)
        fitness_score = anomaly_detection.calculate_fitness(encoder, decoder, generation)
        genome.fitness = int(fitness_score)  # int(np.median(scores))


class NeatOutlier:
    def __init__(self, config, generations=100):
        super().__init__()
        self.generations = generations
        self.encoder = None
        self.decoder = None
        self.config = config
        self.stats = None
        self.winner = None

    def fit(self, saving_path='./logs'):
        population = neat.Population(self.config)
        self.stats = neat.StatisticsReporter()

        population.add_reporter(self.stats)
        population.add_reporter(neat.StdOutReporter(True))

        self.winner = population.run(eval_genomes, self.generations, anomaly_detection, neat, self.config)
        # Save the winner.
        os.makedirs(saving_path + '/winner-AE', exist_ok=True)

        self.stats.save(saving_path)

        with open(saving_path + '/winner-AE' + '/best_model.pkl', 'wb+') as f:
            pickle.dump(self.winner, f)

        self.encoder, self.decoder = neat.nn.FeedForwardNetwork.create_autoencoder(self.winner, self.config)

        return self

    def predict(self, model_path):
        file = open(model_path, mode="rb")
        self.winner = pickle.load(file)
        self.encoder, self.decoder = neat.nn.FeedForwardNetwork.create_autoencoder(self.winner, self.config)
        return self

    # TODO Add checkointing between generations


# Sources:
# https://neat-python.readthedocs.io/en/latest/neat_overview.html
# https://neat-python.readthedocs.io/en/latest/_modules/checkpoint.html
# https://www.programcreek.com/python/example/112263/neat.Checkpointer

if __name__ == '__main__':
    start = datetime.now().strftime("%H:%M:%S-%d/%m/%Y")
    print(f"Program start... {start}")
    print(f"Run UUID: {RUN_UUID}")

    parser = argparse.ArgumentParser(description='Generic runner for Convolutional AE models')
    parser.add_argument('--config', '-c',
                        dest="filename",
                        metavar='FILE',
                        help='path to the config file',
                        default='config/evolve-autoencoder.cfg')
    parser.add_argument('--dataset', '-d',
                        dest="dataset",
                        metavar='FILE',
                        help='path to the dataset file',
                        default='../../datasets/CVD_curriculum.csv')

    parser.add_argument('--curriculum_levels', '-cl',
                        dest="curriculum_levels",
                        metavar='STRING',
                        help='curriculum_levels',
                        default='two')

    args = parser.parse_args()

    """Prepare dataset for anomaly detection based on configuration file"""
    config = AnomalyDetectionConfig(neat.AutoencoderGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                                    neat.DefaultStagnation, args.filename)

    saving_path = f"./logs/{RUN_UUID}/{config.generations}_generations/{args.curriculum_levels}_levels"
    os.makedirs(saving_path, exist_ok=True)

    # Split the data into X levels of difficulty
    data, target = data_loader.curriculum_cvd_dataset(filename=args.dataset, levels=args.curriculum_levels,
                                                      percentage=config.data_percentage)

    # Print config file content
    print(f"Generations: {config.generations}\n"
          f"Population size: {config.pop_size}\n"
          f"Data percentage: {config.data_percentage}\n"
          f"Test size: {config.test_size}\n")

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

    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=config.test_size, random_state=42)

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

    anomaly_detection = AnomalyDetection(X_train, X_test, y_train, y_test, [0], [1], config.generations,
                                         args.curriculum_levels)

    """Adjust number of generations"""
    neat_outlier = NeatOutlier(config, generations=config.generations)

    """Perform neuroevolution on training dataset"""
    NO = neat_outlier.fit(saving_path=saving_path)
    #NO = neat_outlier.predict(f"path_to_best_model\best_model.pkl")

    """Plotting results from anomaly detection on graph"""

    fitness_score, mse_results = anomaly_detection.calculate_roc_auc_curve(neat_outlier.encoder, neat_outlier.decoder)
    print(f"=====================================")
    print(
        f"Model AUC score: {anomaly_detection.AUC} with MSE: {anomaly_detection.MSE} test counter: {anomaly_detection.test_counter}")

    # visualize.save_test_mse_results(mse_results, filename=saving_path + "/test_mse_results.csv")

    visualize.optimal_roc_curve(neat_outlier.encoder, neat_outlier.decoder, anomaly_detection, True,
                                filename=saving_path + "/optimal_roc_curve.svg")

    visualize.plot_roc_curve(anomaly_detection.roc_auc, anomaly_detection.FPR_array, anomaly_detection.TPR_array, True,
                             filename=saving_path + "/roc_curve.svg")
    visualize.plot_metrics(anomaly_detection.metrics, True, filename=saving_path + "/metrics.svg")

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
