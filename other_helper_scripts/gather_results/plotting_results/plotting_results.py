# Copyright (c) OpenMMLab. All rights reserved.
import os

import matplotlib.pyplot as plt

import numpy as np

import argparse

import seaborn as sns
from matplotlib.colors import ListedColormap


from get_overview_results import datasets, versions, mapped_version, dgm_models, metrics, evaluations, get_results, \
    create_results_dict


def get_args():
    args = argparse.ArgumentParser()
    args.add_argument("metric", choices=['BinaryF1', 'WeightedF1', 'AUC'])
    args.add_argument("evaluation", choices=['utility', 'detection'])
    args.add_argument("--postprocessing", action='store_true')
    # args.add_argument("version", type=str, choices=['uncons.', 'random', 'kde', 'corr'])
    # args.add_argument("expdir", type=str)
    return args.parse_args()


def data_test():
    summary_dict = {}
    keys = ['Agent', 'Action', 'Loc', 'Overall']

    summary_dict['Baseline'] = list(map(float, "43.37	24.01	34.81	31.89".split()))
    summary_dict['Freq'] = list(map(float, "45.51059	29.71455	34.85439	35.07".split()))
    summary_dict['Sim'] = list(map(float, "44.09814	28.64174	37.62576	35.04".split()))
    summary_dict['Rev-Freq'] = list(map(float, "37.32258	28.28013	39.79178	33.85".split()))
    summary_dict['Rev-Sim'] = list(map(float, "38.92118	28.23563	37.33061	33.5".split()))
    summary_dict['Rnd'] = list(map(float, "41.24085	27.84367	34.57371	33.08".split()))

    return summary_dict
    


def bar_plot(args, ax, data, colors=None, total_width=0.8, single_width=1, legend=True):
    # adapted code from https://stackoverflow.com/questions/14270391/python-matplotlib-multiple-bars
    """Draws a bar plot with multiple bars per data point.

    Parameters
    ----------
    ax : matplotlib.pyplot.axis
        The axis we want to draw our plot on.

    data: dictionary
        A dictionary containing the data we want to plot. Keys are the names of the
        data, the items is a list of the values.

        Example:
        data = {
            "x":[1,2,3],
            "y":[1,2,3],
            "z":[1,2,3],
        }

    colors : array-like, optional
        A list of colors which are used for the bars. If None, the colors
        will be the standard matplotlib color cyle. (default: None)

    total_width : float, optional, default: 0.8
        The width of a bar group. 0.8 means that 80% of the x-axis is covered
        by bars and 20% will be spaces between the bars.

    single_width: float, optional, default: 1
        The relative width of a single bar within a group. 1 means the bars
        will touch eachother within a group, values less than 1 will make
        these bars thinner.

    legend: bool, optional, default: True
        If this is set to true, a legend will be added to the axis.
    """

    # Check if colors where provided, otherwhise use the default color cycle
    # color_palette0 = sns.color_palette("Reds", 10)
    # color_palette1 = sns.color_palette("Blues", 70)
    # color_palette2 = sns.color_palette("Greens", 70)
    # colors = sns.color_palette(
    #     list(reversed(color_palette0[1::2][:2])) + color_palette1[20::15][:2] + color_palette2[20::15][:2])

    # color_palette00 = sns.color_palette("Greys", 8)
    # color_palette0 = sns.color_palette("Purples", 6)
    # color_palette1 = sns.color_palette("Blues", 70)
    # color_palette2 = sns.color_palette("Greens", 70)
    # colors = sns.color_palette(
    #     [color_palette00[2]] + [color_palette0[2]] + color_palette1[20::15][:2] + color_palette2[20::15][:2])
    color_palette0 = sns.color_palette("Reds", 6)

    color_palette1 = sns.color_palette("Blues", 50)
    color_palette2 = sns.color_palette("Greens", 50)

    # color_palette = sns.color_palette("PuBuGn", 50)
    colors = sns.color_palette(list(reversed(color_palette0[1::2][:2])) + color_palette1[20::15][:2] + color_palette2[20::15][:2])
    # # color_palette = ListedColormap(color_palette.as_hex())



    # Number of bars per group
    n_bars = len(data)

    # The width of a single bar
    bar_width = total_width / n_bars

    # List containing handles for the drawn bars, used for the legend
    bars = []

    # Iterate over all data
    order_to_plot = versions #['Baseline', 'Rnd', 'Rev-Sim', 'Sim', 'Rev-Freq', 'Freq']
    patterns = [ "//////" , "\\\\\\\\\\" , "||||" , "-----" , "++++" , "xxxxx"] #, "o", "O", ".", "*" ]
    for i,name in enumerate(order_to_plot):
    # for i, (name, values) in enumerate(data.items()):
        values = data[name]
        # The offset in x direction of that bar
        x_offset = (i - n_bars / 2) * bar_width + bar_width / 2

        # Draw a bar for every value of that type
        for x, y in enumerate(values):
            bar = ax.bar(x + x_offset, y, width=bar_width * single_width, edgecolor=colors[i], hatch=patterns[i], color='w')

        # Add a handle to the last drawn bar, which we'll need for the legend
        bars.append(bar[0])

    # Draw legend if we need
    if legend:
        ax.legend(bars, order_to_plot, loc='lower right', fancybox=True, framealpha=0.9) #loc=(.76,0.0)

    ax.set_ylabel('f-mAP')
    # ax.set_title('Validation set performance')
    plt.xticks(range(4), dgm_models)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig('figs/'+f'{args.metric}_bar_performance.png', format='png', dpi=500)




def main():
    args = get_args()
    metric = args.metric

    if args.postprocessing:
        model_types = ['wgan', 'tablegan']
    else:
        model_types = ['wgan', 'tablegan', 'ctgan']

    summary_dict = {e:[] for e in model_types}
    results = {evaluation: {model_type: {metric:{} for metric in metrics} for model_type in model_types} for evaluation in evaluations}
    for evaluation in evaluations:
        for model_type in model_types:
            # for metric in metrics:
                temp, stddevs = get_results(model_type, metric)

                rdict = create_results_dict(temp)
                summary_dict[model_type] = rdict

    summary_dict = data_test(rdict)
    print(summary_dict)
    # plotting(summary_dict)
    fig, ax = plt.subplots(figsize=(6,4))
    bar_plot(args, ax, summary_dict, total_width=.8, single_width=.9)
    plt.show()

if __name__ == '__main__':
    main()
