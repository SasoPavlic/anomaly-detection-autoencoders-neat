from __future__ import print_function

import copy
import warnings
import seaborn as sns
import matplotlib
from matplotlib.widgets import Slider
import graphviz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve


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
                connections.add((cg.in_node_id, cg.out_node_id))

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
                connections.add((cg.in_node_id, cg.out_node_id))

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


def plot_slider(inputs, encoder, decoder, view=False, filename='./logs/slider.png'):
    """ Plots the slider for encoder and decoder. """
    input = inputs[0]
    bottleneck_output = encoder.activate(input)
    reconstructed = decoder.activate(bottleneck_output)

    fig, ax = plt.subplots()
    im = ax.imshow(np.array([reconstructed]))

    axcolor = 'lightgoldenrodyellow'
    decoder_input_axes = []
    decoder_input_sliders = []
    for i, out in enumerate(bottleneck_output):
        y_pos = (i + 1) * 0.1
        decoder_input_axes.append(plt.axes([0.25, y_pos, 0.65, 0.03], facecolor=axcolor))
        decoder_input_sliders.append(Slider(decoder_input_axes[i], f'bottleneck_input{str(i)}', 0.0, 1.0, valinit=out))

    def create_update_func(i):
        def update(val):
            bottleneck_output[i] = val
            print(bottleneck_output)
            im.set_data(np.array([decoder.activate(bottleneck_output)]))

        return update

    for i, slider in enumerate(decoder_input_sliders):
        slider.on_changed(create_update_func(i))

    if filename is not None:
        plt.savefig(filename)

    if view:
        plt.show()
        plt.close()
        fig = None

    return fig


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


def plot_roc_curve(y_test, FPR_array, TPR_array, view=False, filename='./logs/roc_curve.png'):
    # https://www.analyticsvidhya.com/blog/2020/06/auc-roc-curve-machine-learning/
    random_probs = [0 for i in range(len(y_test))]
    p_fpr, p_tpr, thresholds = roc_curve(y_test, random_probs, pos_label=1)

    # This is the ROC curve
    plt.style.use('seaborn')

    # plot roc curves
    plt.plot(FPR_array, TPR_array, linestyle='--', color='green', label='Autoencoder')
    plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')

    plt.title(f'ROC curve - AUC: {round(np.trapz(TPR_array, FPR_array), 3)}')
    # x label
    plt.xlabel('False Positive Rate')
    # y label
    plt.ylabel('True Positive rate')
    plt.legend(loc='best')
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
