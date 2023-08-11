from __future__ import print_function

import copy
import csv
import math
import warnings

import graphviz
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve, auc


def save_test_mse_results(results, filename='./logs/test_mse_results.csv'):
    df = pd.DataFrame(results, columns=['MSE'])
    df.to_csv(filename, index=False)

def plot_stats(statistics, ylog=False, view=False, filename='./logs/avg_fitness.svg'):
    """ Plots the population's average and best fitness. """
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return

    generation = range(len(statistics.most_fit_genomes))
    best_fitness = [c.fitness for c in statistics.most_fit_genomes]
    avg_fitness = np.array(statistics.get_fitness_mean())
    stdev_fitness = np.array(statistics.get_fitness_stdev())

    plt.plot(generation, avg_fitness, 'b-', label="average")
    plt.plot(generation, avg_fitness - stdev_fitness, 'g-.', label="-1 sd")
    plt.plot(generation, avg_fitness + stdev_fitness, 'g-.', label="+1 sd")
    plt.plot(generation, best_fitness, 'r-', label="best")

    plt.title("Population's average and best fitness")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.grid()
    plt.legend(loc="best")
    if ylog:
        plt.gca().set_yscale('symlog')

    plt.savefig(filename)
    if view:
        plt.show()

    plt.close()


def plot_species(statistics, view=False, filename='./logs/speciation.svg'):
    """ Visualizes speciation throughout evolution. """
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return

    species_sizes = statistics.get_species_sizes()
    num_generations = len(species_sizes)
    curves = np.array(species_sizes).T

    fig, ax = plt.subplots()
    ax.stackplot(range(num_generations), *curves)

    plt.title("Speciation")
    plt.ylabel("Size per Species")
    plt.xlabel("Generations")

    plt.savefig(filename)

    if view:
        plt.show()

    plt.close()


def draw_net_encoder(config, genome, view=False, filename='logs/encoder', node_names=None, show_disabled=True,
                     prune_unused=False,
                     node_colors=None, fmt='svg', genome_config=None):
    """ Receives a genome and draws a neural network with arbitrary topology. """

    # For Windows OS (location of Graphviz installation)
    import os
    if os.name == 'nt':
        os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'

    if genome_config is None:
        genome_config = config.genome_config

    # Attributes for network nodes.
    if graphviz is None:
        warnings.warn("This display is not available due to a missing optional dependency (graphviz)")
        return

    if node_names is None:
        node_names = {}

    assert type(node_names) is dict

    if node_colors is None:
        node_colors = {}

    assert type(node_colors) is dict

    node_attrs = {
        'shape': 'circle',
        'fontsize': '9',
        'height': '0.2',
        'width': '0.2'}

    dot = graphviz.Digraph(format=fmt, node_attr=node_attrs)

    inputs = set()
    for k in genome_config.encoder_input_keys:
        inputs.add(k)
        name = node_names.get(k, str(k))
        input_attrs = {'style': 'filled', 'shape': 'box', 'fillcolor': node_colors.get(k, 'lightgray')}
        dot.node(name, _attributes=input_attrs)

    outputs = set()
    for k in genome_config.encoder_output_keys:
        outputs.add(k)
        name = node_names.get(k, str(k))
        node_attrs = {'style': 'filled', 'fillcolor': node_colors.get(k, 'lightblue')}

        dot.node(name, _attributes=node_attrs)

    if prune_unused:
        connections = set()
        for cg in genome.connections.values():
            if cg.enabled or show_disabled:
                connections.add((cg.key[0], cg.key[1]))

        used_nodes = copy.copy(outputs)
        pending = copy.copy(outputs)
        while pending:
            new_pending = set()
            for a, b in connections:
                if b in pending and a not in used_nodes:
                    new_pending.add(a)
                    used_nodes.add(a)
            pending = new_pending
    else:
        used_nodes = set(genome.nodes.keys())

    for n in used_nodes:
        if n in inputs or n in outputs:
            continue

        attrs = {'style': 'filled', 'fillcolor': node_colors.get(n, 'white')}
        dot.node(str(n), _attributes=attrs)

    for cg in genome.connections.values():
        if cg.enabled or show_disabled:
            # if cg.input not in used_nodes or cg.output not in used_nodes:
            #    continue
            input, output = cg.key
            a = node_names.get(input, str(input))
            b = node_names.get(output, str(output))
            style = 'solid' if cg.enabled else 'dotted'
            color = 'green' if cg.weight > 0 else 'red'
            width = str(0.1 + abs(cg.weight / 5.0))
            dot.edge(a, b, _attributes={'style': style, 'color': color, 'penwidth': width})

    dot.render(filename, view=view)

    return dot


def draw_net_decoder(config, genome, view=False, filename='logs/decoder', node_names=None, show_disabled=True,
                     prune_unused=False,
                     node_colors=None, fmt='svg', genome_config=None):
    """ Receives a genome and draws a neural network with arbitrary topology. """

    # For Windows OS (location of Graphviz installation)
    import os
    if os.name == 'nt':
        os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'

    if genome_config is None:
        genome_config = config.genome_config

    # Attributes for network nodes.
    if graphviz is None:
        warnings.warn("This display is not available due to a missing optional dependency (graphviz)")
        return

    if node_names is None:
        node_names = {}

    assert type(node_names) is dict

    if node_colors is None:
        node_colors = {}

    assert type(node_colors) is dict

    node_attrs = {
        'shape': 'circle',
        'fontsize': '9',
        'height': '0.2',
        'width': '0.2'}

    dot = graphviz.Digraph(format=fmt, node_attr=node_attrs)

    inputs = set()
    for k in genome_config.decoder_input_keys:
        inputs.add(k)
        name = node_names.get(k, str(k))
        input_attrs = {'style': 'filled', 'shape': 'box', 'fillcolor': node_colors.get(k, 'lightgray')}
        dot.node(name, _attributes=input_attrs)

    outputs = set()
    for k in genome_config.decoder_input_keys:
        outputs.add(k)
        name = node_names.get(k, str(k))
        node_attrs = {'style': 'filled', 'fillcolor': node_colors.get(k, 'lightblue')}

        dot.node(name, _attributes=node_attrs)

    if prune_unused:
        connections = set()
        for cg in genome.connections.values():
            if cg.enabled or show_disabled:
                connections.add((cg.key[0], cg.key[1]))

        used_nodes = copy.copy(outputs)
        pending = copy.copy(outputs)
        while pending:
            new_pending = set()
            for a, b in connections:
                if b in pending and a not in used_nodes:
                    new_pending.add(a)
                    used_nodes.add(a)
            pending = new_pending
    else:
        used_nodes = set(genome.nodes.keys())

    for n in used_nodes:
        if n in inputs or n in outputs:
            continue

        attrs = {'style': 'filled', 'fillcolor': node_colors.get(n, 'white')}
        dot.node(str(n), _attributes=attrs)

    for cg in genome.connections.values():
        if cg.enabled or show_disabled:
            # if cg.input not in used_nodes or cg.output not in used_nodes:
            #    continue
            input, output = cg.key
            a = node_names.get(input, str(input))
            b = node_names.get(output, str(output))
            style = 'solid' if cg.enabled else 'dotted'
            color = 'green' if cg.weight > 0 else 'red'
            width = str(0.1 + abs(cg.weight / 5.0))
            dot.edge(a, b, _attributes={'style': style, 'color': color, 'penwidth': width})

    print(os.getcwd())
    dot.render(filename, view=view)

    return dot


def plot_metrics(metrics, view=False, filename='./logs/metrics.svg'):
    list_accuracy = list()
    list_F1 = list()
    list_recall = list()
    list_precision = list()
    quantiles = list()

    for metric in metrics:
        list_accuracy.append(metric.accuracy)
        list_F1.append(metric.F1)
        list_recall.append(metric.recall)
        list_precision.append(metric.precision)
        quantiles.append(metric.quantile)

    # plotting the points
    plt.plot(quantiles, list_accuracy, label="Accuracy")
    plt.plot(quantiles, list_F1, label="F1-measure")
    plt.plot(quantiles, list_recall, label="Recall")
    plt.plot(quantiles, list_precision, label="Precision ")

    # naming the x axis
    plt.xlabel('Quantiles')
    # naming the y axis
    plt.ylabel('Score')

    # giving a title to my graph
    plt.title('Effectiveness of anomaly detection')

    plt.legend()

    # function to show the plot
    plt.savefig(filename)
    if view:
        plt.show()

    plt.close()


def optimal_roc_curve(encoder, decoder, anomalyDetection, view=False, filename='./logs/optimal_roc_curve.png'):
    decoded_instances, scores, targets = anomalyDetection.calculate_final_mse(encoder, decoder)
    fpr, tpr, thresholds = roc_curve(targets, scores)
    roc_auc = auc(fpr, tpr)

    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    y_pred = [0 if x <= optimal_threshold else 1 for x in scores]

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='Autoencoder (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--', label='Random classifier (AUC = 0.50)')
    plt.plot([fpr[optimal_idx], tpr[optimal_idx]], [fpr[optimal_idx], tpr[optimal_idx]], color='red', lw=lw,
             linestyle=':',
             label=f'Distance = {round(math.dist([0, 1], [fpr[optimal_idx], tpr[optimal_idx]]), 2)}')
    plt.plot(fpr[optimal_idx], tpr[optimal_idx], '-ro',
             label=f'Optimal threshold (FPR={round(fpr[optimal_idx], 2)}, TPR={round(tpr[optimal_idx], 2)})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    # plt.title("Performance of the optimal AE model for AD on fault detection dataset.")
    plt.legend(loc='lower right')
    plt.savefig(filename)

    if view:
        plt.show()

    plt.close()


def plot_roc_curve(roc_auc, FPR_array, TPR_array, view=False, filename='./logs/roc_curve.png'):
    plt.figure()
    lw = 2
    plt.plot(
        FPR_array[1],
        TPR_array[1],
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.2f)" % roc_auc[1],
    )

    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic")
    plt.legend(loc="lower right")
    plt.savefig(filename)

    if view:
        plt.show()

    plt.close()


def plot_heatmap(dataset, view=False, filename='./logs/heatmap.png'):
    df = pd.DataFrame(data=np.c_[dataset['data'], dataset['target']], columns=dataset['feature_names'] + ['target'])
    print(df.isna().sum())

    corr = df.corr()
    matplotlib.pyplot.subplots(figsize=(15, 10))
    sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True,
                cmap=sns.diverging_palette(220, 20, as_cmap=True))

    df.describe()
    plt.savefig(filename)
    if view:
        plt.show()

    plt.close()
