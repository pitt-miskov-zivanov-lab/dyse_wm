import os
import re
import argparse
import glob
import math
from collections import defaultdict
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random


def get_traces(input_file: str, default_levels=3):
    """Load simulation values from a trace file

    Trace file should contain (optionally) individual runs, followed by the
    frequency summaries, followed by the squares summaries for each element
    (output_format 1 or 3 from the simulator)
    """

    # TODO: move to traces.py

    # dictionary to store traces for each element
    trace_data = defaultdict(lambda: defaultdict())

    # read traces from trace file
    with open(input_file) as in_file:
        trace_file_lines = in_file.readlines()
    trace_file_lines = [x.strip() for x in trace_file_lines]

    run = 0
    frequency_summary = False
    squares_summary = False
    for trace_line_content in trace_file_lines:

        # check each line for the Run #, Frequency/Squares summaries, or values
        if re.match(r'Run #[0-9]+', trace_line_content):
            # save the last run # as the total number of runs
            run = re.findall(r'Run #([0-9]+)', trace_line_content)[0]

        elif re.match('Frequency Summary:', trace_line_content):
            # set frequency summary flag
            frequency_summary = True

        elif re.match('Squares Summary:', trace_line_content):
            # set squares summary flag
            squares_summary = True

        else:
            # get trace values and element name
            trace_values_str = trace_line_content.split(' ')
            element_name = trace_values_str[0].split('|')[0]

            if element_name != '':
                # get num states for this element if available
                if len(trace_values_str[0].split('|'))>1:
                    levels = int(trace_values_str[0].split('|')[1])
                else:
                    levels = default_levels

                # get simulation values
                trace_values = [float(val) for val in trace_values_str[1:]]

                # store trace data
                trace_data[element_name]['levels'] = levels
                # will save the number read from the last occurrence of "Run#" in the file
                # adding 1 to correct for zero indexed run number
                trace_data[element_name]['runs'] = int(run) + 1
                if squares_summary:
                    trace_data[element_name]['squares'] = trace_values
                    # calculate avg and stdev values
                    runs = trace_data[element_name]['runs']
                    levels = trace_data[element_name]['levels']
                    freq_vals = trace_data[element_name]['frequency']
                    avg_vals = [float(val)/runs for val in freq_vals]
                    stdev_vals = [np.sqrt(float(val)/runs - avg_vals[idx]**2) for idx, val in enumerate(trace_values)]
                    # vals_pool_transpose = list(trace_data[element_name]['traces'].values())
                    # vals_pool = list(map(list, zip(*vals_pool_transpose)))

                    trace_data[element_name]['avg'] = avg_vals
                    trace_data[element_name]['stdev'] = stdev_vals
                    trace_data[element_name]['avg_percent'] = [100*val/(levels-1) for val in avg_vals]
                    trace_data[element_name]['stdev_percent'] = [100*val/(levels-1) for val in stdev_vals]
                    # trace_data[element_name]['ci'] = list(map(bootstrap, vals_pool))

                elif frequency_summary:
                    trace_data[element_name]['frequency'] = trace_values
                else:
                    if 'traces' in trace_data[element_name]:
                        trace_data[element_name]['traces'].update({run : trace_values})
                    else:
                        trace_data[element_name]['traces'] = {run : trace_values}

    return trace_data

def bootstrap(data):
    """
    Generate bootstrap samples, evaluating `func` at each resampling. `bootstrap` returns a function,
    which can be called to obtain confidence intervals of interest.

    Note that func, resample_size, resample_times and intervals can all be adjusted.
    """
    func=np.mean
    resample_size = [int(len(data)*0.25), int(len(data)*0.30)]
    resample_times = int(len(data)*0.10)
    intervals = 0.95

    simulations = list()
    for c in range(resample_times):
        itersample = np.random.choice(data, size=random.randint(resample_size[0], resample_size[1]), replace=True)
        simulations.append(func(itersample))
    simulations.sort()

    def ci(p):
        """
        Return 2-sided symmetric confidence interval specified by p.
        """
        u_pval = (1+p)/2.
        l_pval = (1-u_pval)
        l_indx = int(np.floor(resample_times*l_pval))
        u_indx = int(np.floor(resample_times*u_pval))
        return [simulations[l_indx],simulations[u_indx]]

    return(ci(intervals))


def get_end_values(trace_data: dict(), elements_list=[''], normalize_levels=True):
    """Get end of simulation values"""
    # TODO: move to traces.py

    if elements_list == ['']:
        # assuming same elements in each scenario
        elements_list = trace_data.keys()

    end_values = {element : trace_data[element]['avg_percent'][-1] if normalize_levels
            else trace_data[element]['avg'][-1] for element in elements_list}

    return end_values


def plot_heatmap(
        trace_data: dict(),
        elements_list=[''],
        normalize_levels=True,
        timesteps_list=[],
        timesteps_labels=[[],['']],
        z_limits=['default'],
        x_label='Step',
        y_label='Run',
        z_label='Level',
        cmap='gray',
        style='whitegrid',
        fig_size=(6,4)):
    """Generate heatmaps showing simulation values of individual runs

    Note that the trace file must have individual runs, not just summaries
    (output_format 1 from the simulator)
    """

    heatmap_plots = defaultdict()

    if elements_list == ['']:
        elements_list = trace_data.keys()

    for element in elements_list:
        this_element_data = trace_data[element]

        levels = this_element_data['levels']

        this_element_runs = this_element_data.get('traces')

        if this_element_runs is None:
            raise ValueError('Bad trace format, missing individual runs')

        # process each run for plotting
        heatmap_data = []
        for run_index, run_values in this_element_runs.items():

            if normalize_levels:
                # create a list of simulation values as percentages
                run_values = [100*float(val)/(levels-1) for val in run_values]

            # get only the specified time steps
            timesteps = list(range(len(run_values)))
            if timesteps_list != []:
                timesteps = [timesteps[int(step)] for step in timesteps_list]
                run_values = [run_values[int(step)] for step in timesteps_list]

            # store data for plotting
            heatmap_data += [tuple(run_values)]

        # generate plot
        sns.set_style(style)
        fig, ax = plt.subplots(figsize=fig_size)

        if z_limits[0] == 'default':
            # assuming all data from the same element have the same levels
            vmin = 0
            if normalize_levels:
                vmax = 100
            else:
                vmax = levels
        elif z_limits[0] == 'auto':
            pass
        elif len(z_limits) != 2:
            raise ValueError('Invalid z_limits input, must be [<min>, <max>]')
        else:
            # use input z_limits
            vmin = z_limits[0]
            vmax = z_limits[1]

        heatmap_plot = sns.heatmap(heatmap_data, cmap=cmap, cbar_kws={'label': z_label}, vmin=0, vmax=vmax, ax=ax)

        # set xticks and labels if provided
        plt.yticks(rotation=0)
        if timesteps_list != []:
            plt.xticks(timesteps_list)
            ax.set_xlim([timesteps_list[0], timesteps_list[-1]])
        if timesteps_labels != [[], ['']]:
            plt.xticks(timesteps_labels[0], timesteps_labels[1], rotation='vertical')

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(element)

        plt.tight_layout()
        fig.tight_layout()

        heatmap_plots[element] = fig

        plt.close()

    return heatmap_plots


def plot_each_run(
        trace_data: dict(),
        elements_list=[''],
        normalize_levels=True,
        timesteps_list=[],
        timesteps_labels=[[],['']],
        y_limits=['default'],
        x_label='Step',
        y_label='Level',
        style='whitegrid',
        linewidth=1,
        fig_size=(5,3)):
    """Plot each run as its own line on the same plot axes"""

    run_plots = defaultdict()

    if elements_list == ['']:
        elements_list = trace_data.keys()

    for element in elements_list:
        this_element_data = trace_data[element]

        levels = this_element_data['levels']

        this_element_runs = this_element_data.get('traces')

        if this_element_runs is None:
            raise ValueError('Bad trace format, missing individual runs')

        # process each run for plotting
        plot_data = []
        for run_index, run_values in this_element_runs.items():

            if normalize_levels:
                # create a list of simulation values as percentages
                run_values = [100*float(val)/(levels-1) for val in run_values]

            # get only the specified time steps
            timesteps = list(range(len(run_values)))
            if timesteps_list != []:
                timesteps = [timesteps[int(step)] for step in timesteps_list]
                run_values = [run_values[int(step)] for step in timesteps_list]

            # TODO: use colormap values for line colors
            plot_data += [dict(x=timesteps, y=run_values, label=element, linewidth=linewidth)]

        # generate plot
        sns.set_style(style)
        fig, ax = plt.subplots(figsize=fig_size)

        [plt.plot(d['x'], d['y'], label=d['label'], linewidth=d['linewidth']) for d in plot_data]

        # using x range of [0, steps-1] to plot initial value at 0
        ax.set_xlim([0, len(plot_data[0]['x'])-1])

        set_y_limits(ax, y_limits, levels, normalize_levels)

        # set xticks and labels if provided
        if timesteps_list != []:
            plt.xticks(timesteps_list)
            ax.set_xlim([timesteps_list[0], timesteps_list[-1]])
        if timesteps_labels != [[], ['']]:
            plt.xticks(timesteps_labels[0], timesteps_labels[1], rotation='vertical')

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        ax.set_title(element)

        plt.tight_layout()
        fig.tight_layout()

        run_plots[element] = fig

        plt.close()

    return run_plots


def get_value_counts(element_trace_data, levels):
    """Calculate distribution of values at each time step"""

    # format element trace data
    element_trace_data_array = np.array(list(element_trace_data.values()))

    # get value counts at each time step across runs
    # transposing to get count across time step
    # TODO: calculate bins accounting for space between levels
    bins = [0] + [x*1.01 for x in range(levels)]
    value_counts = np.array([np.array(np.histogram(x, bins)[0]) for x in element_trace_data_array.T])

    return value_counts


def plot_distribution(
        trace_data: dict(),
        elements_list=[''],
        normalize_levels=False,
        timesteps_list=[],
        timesteps_labels=[[],['']],
        x_label='Step',
        y_label='Value Count',
        colors=None,
        style='whitegrid',
        fig_size=(5,3)):
    """Plot distribution of values at each time step

    Note that the trace file must have individual runs, not just summaries
    (output_format 1 from the simulator)
    """

    distribution_plots = defaultdict()

    if elements_list == ['']:
        elements_list = trace_data.keys()

    for element in elements_list:
        this_element_data = trace_data[element]

        levels = int(this_element_data['levels'])

        if colors is not None and len(colors) != levels:
            raise ValueError('Length of colors must equal number of levels')

        this_element_runs = this_element_data.get('traces')

        if this_element_runs is None:
            raise ValueError('Bad trace format, missing individual runs')

        # process trace data for plotting
        value_counts = get_value_counts(this_element_runs, levels)

        # specified timesteps
        # get only the specified time steps
        timesteps = list(range(value_counts.shape[0]))
        if timesteps_list != []:
            timesteps = [timesteps[int(step)] for step in timesteps_list]
            value_counts = np.array([value_counts[int(step),:] for step in timesteps_list])

        # generate plots
        sns.set_style(style)
        fig, ax = plt.subplots(figsize=fig_size)

        bar_width = 1
        prev_vals = value_counts[:,0]
        plt.bar(timesteps, prev_vals, bar_width, label=0, color=colors[0] if colors is not None else None)

        for level in range(levels)[1:]:
            curr_vals = value_counts[:, level]
            plt.bar(timesteps, curr_vals, bar_width, bottom=prev_vals, linewidth=0, label=level,
                    color=colors[level] if colors is not None else None)

            prev_vals = curr_vals

        # add legend
        handles, labels = ax.get_legend_handles_labels()
        if normalize_levels:
            labels = [100*float(val)/(levels-1) for val in labels]
        plt.legend(handles, labels, frameon=False)

        # set xticks and labels if provided
        plt.yticks(rotation=0)
        if timesteps_list != []:
            plt.xticks(timesteps_list)
            ax.set_xlim([timesteps_list[0], timesteps_list[-1]])
        if timesteps_labels != [[], ['']]:
            plt.xticks(timesteps_labels[0], timesteps_labels[1], rotation='vertical')

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(element)

        plt.tight_layout()
        fig.tight_layout()

        distribution_plots[element] = fig

        plt.close()

    return distribution_plots


def plot_average_multiple_elements(
        trace_data: dict(),
        elements_list=[''],
        normalize_levels=True,
        errorbars=False,
        timesteps_list=[],
        timesteps_labels=[[],['']],
        y_limits=['default'],
        x_label='Step',
        y_label='Level',
        style='whitegrid',
        linewidth=1,
        colors=None,
        linestyles=None,
        fig_size=(5,3)):
    """Plot multiple elements on the same axes"""

    if colors is not None and len(elements_list) != len(colors):
        raise ValueError('Length of colors must equal length of elements_list')

    if linestyles is not None and len(elements_list) != len(linestyles):
        raise ValueError('Length of linestyles must equal length of elements_list')

    if elements_list == ['']:
        # assuming same elements in each scenario
        elements_list = trace_data.keys()

    elements_plot_data = []
    for element_index, element in enumerate(elements_list):
        element_data = trace_data[element]

        levels = int(element_data['levels'])

        if normalize_levels:
            # stdev as percentage
            stdev_vals = element_data['stdev_percent']
            # average level as percentage
            avg_vals = element_data['avg_percent']
        else:
            stdev_vals = element_data['stdev']
            avg_vals = element_data['avg']

        # get only the specified time steps
        timesteps = list(range(len(avg_vals)))
        if timesteps_list != []:
            timesteps = [timesteps[int(step)] for step in timesteps_list]
            avg_vals = [avg_vals[int(step)] for step in timesteps_list]
            stdev_vals = [stdev_vals[int(step)] for step in timesteps_list]

        # save data and plotting information
        elements_plot_data += [dict(
                x=timesteps,
                y=avg_vals,
                label=element,
                error=stdev_vals,
                levels=levels,
                linewidth=linewidth,
                color=colors[element_index] if colors is not None else None,
                linestyle=linestyles[element_index] if linestyles is not None else None)]

    # generate plot
    sns.set_style(style)
    fig, ax = plt.subplots(figsize=fig_size)

    # plot each trace
    [single_plot(d, errorbars) for d in elements_plot_data]

    # using x range of [0, steps-1] to plot initial value at 0
    ax.set_xlim([0, len(elements_plot_data[0]['x'])-1])

    set_y_limits(ax, y_limits, levels, normalize_levels)

    # set xticks and labels if provided
    if timesteps_list != []:
        plt.xticks(timesteps_list)
        ax.set_xlim([timesteps_list[0], timesteps_list[-1]])
    if timesteps_labels != [[],['']]:
        plt.xticks(timesteps_labels[0], timesteps_labels[1], rotation='vertical')

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # add legend
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles, labels, frameon=False)

    plt.tight_layout()
    fig.tight_layout()

    elements_plot = fig

    plt.close()

    return elements_plot


def plot_average(
        trace_data_list: list(),
        elements_list=[''],
        normalize_levels=True,
        scenario_labels=[''],
        errorbars=False,
        timesteps_list=[],
        timesteps_labels=[[],['']],
        y_limits=['default'],
        x_label='Step',
        y_label='Level',
        style='whitegrid',
        linewidth=1,
        colors=None,
        linestyles=None,
        fig_size=(5,3)):
    """Plot average traces

    If trace_data_list has more than one item, plots each as a separate scenario
    """

    if not isinstance(trace_data_list, list):
        raise ValueError('Input trace_data_list must be a list')

    if scenario_labels == ['']:
        scenario_labels = [f'Scenario {idx}' for idx, _ in enumerate(trace_data_list)]

    # check length of inputs
    if len(trace_data_list) != len(scenario_labels):
        raise ValueError('Length of scenario_labels must equal length of trace_data_list')

    if colors is not None and len(trace_data_list) != len(colors):
        raise ValueError('Length of colors must equal length of trace_data_list')

    if linestyles is not None and len(trace_data_list) != len(linestyles):
        raise ValueError('Length of linestyles must equal length of trace_data_list')

    # will get traces from each element across scenarios
    avg_plots = defaultdict()

    if elements_list == ['']:
        # assuming same elements in each scenario
        elements_list = trace_data_list[0].keys()

    for element in elements_list:
        plot_data = []
        all_element_data = [trace_data[element] for trace_data in trace_data_list]

        for scenario_index, scenario_data in enumerate(all_element_data):

            levels = int(scenario_data['levels'])

            if normalize_levels:
                # stdev as percentage
                stdev_vals = scenario_data['stdev_percent']
                # average level as percentage
                avg_vals = scenario_data['avg_percent']
            else:
                stdev_vals = scenario_data['stdev']
                avg_vals = scenario_data['avg']

            # get only the specified time steps
            timesteps = list(range(len(avg_vals)))
            if timesteps_list != []:
                timesteps = [timesteps[int(step)] for step in timesteps_list]
                avg_vals = [avg_vals[int(step)] for step in timesteps_list]
                stdev_vals = [stdev_vals[int(step)] for step in timesteps_list]

            # save data and plotting information
            plot_data += [dict(
                    x=timesteps,
                    y=avg_vals,
                    label=scenario_labels[scenario_index],
                    error=stdev_vals,
                    levels=levels,
                    linewidth=linewidth,
                    color=colors[scenario_index] if colors is not None else None,
                    linestyle=linestyles[scenario_index] if linestyles is not None else None)]

        # generate plots
        sns.set_style(style)
        fig, ax = plt.subplots(figsize=fig_size)

        # plot each trace
        [single_plot(d, errorbars) for d in plot_data]

        # using x range of [0, steps-1] to plot initial value at 0
        ax.set_xlim([0, len(plot_data[0]['x'])-1])

        set_y_limits(ax, y_limits, levels, normalize_levels)

        # set xticks and labels if provided
        if timesteps_list != []:
            plt.xticks(timesteps_list)
            ax.set_xlim([timesteps_list[0], timesteps_list[-1]])
        if timesteps_labels != [[],['']]:
            plt.xticks(timesteps_labels[0], timesteps_labels[1], rotation='vertical')

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(element)

        # add legend if more than 1 scenario
        if len(trace_data_list) > 1:
            handles, labels = ax.get_legend_handles_labels()
            plt.legend(handles, labels, frameon=False)

        plt.tight_layout()
        fig.tight_layout()

        avg_plots[element] = fig

        plt.close()

    return avg_plots


def plot_compared_average(
        trace_data_list: list(),
        elements_list=[''],
        normalize_levels=True,
        scenario_labels=[''],
        comparison='difference',
        timesteps_list=[],
        timesteps_labels=[[],['']],
        y_limits=['default'],
        x_label='Step',
        y_label='Compared Level',
        style='whitegrid',
        linewidth=1,
        colors=None,
        linestyles=None,
        fig_size=(5,3)):
    """Compare average traces across scenarios

    Inputs:
        comparison: ['difference', 'division', 'percentchange']
                difference - plot difference from first scenario values
                division - divide by first scenario values
                percentchange - plot percent change from first scenario

    """

    # check inputs
    compare_opts = ['difference', 'division', 'percentchange']
    compare_opts_str = ', '.join([opt for opt in compare_opts])
    if comparison not in compare_opts:
        raise ValueError(f'Invalid comparison input, options are: {compare_opts_str}')

    if len(trace_data_list) == 1:
        raise ValueError('Cannot compare with only one set of traces')

    if scenario_labels == ['']:
        # construct trace labels based on comparison input
        if comparison == 'difference':
            scenario_labels = [f'Scenario {idx}-Scenario 0' for idx, _ in enumerate(trace_data_list)][1:]
        elif comparison == 'division':
            scenario_labels = [f'Scenario {idx}/Scenario 0' for idx, _ in enumerate(trace_data_list)][1:]
        elif comparison == 'percentchange':
            scenario_labels = [f'(Scenario {idx}-Scenario 0)/Scenario 0' for idx, _ in enumerate(trace_data_list)][1:]

    if len(scenario_labels) != len(trace_data_list) - 1:
            raise ValueError('Number of scenario labels must be length of trace_data_list - 1')

    if colors is not None and len(colors) != len(trace_data_list) - 1:
        raise ValueError('Length of colors must be length of trace_data_list - 1')

    if linestyles is not None and len(linestyles) != len(trace_data_list) - 1:
        raise ValueError('Length of linestyles must be length of trace_data_list - 1')

    # will get traces from each element across scenarios
    comparison_plots = defaultdict()

    if elements_list == ['']:
        # assuming same elements in each scenario
        elements_list = trace_data_list[0].keys()

    for element in elements_list:
        compared_plot_data = []
        all_element_data = [trace_data[element] for trace_data in trace_data_list]

        for scenario_index, scenario_data in enumerate(all_element_data):

            levels = int(scenario_data['levels'])

            if normalize_levels:
                # average level as percentage
                avg_vals = scenario_data['avg_percent']
            else:
                avg_vals = scenario_data['avg']

            # get only the specified time steps
            timesteps = list(range(len(avg_vals)))
            if timesteps_list != []:
                timesteps = [timesteps[int(step)] for step in timesteps_list]
                avg_vals = [avg_vals[int(step)] for step in timesteps_list]

            if scenario_index == 0:
                # store first scenario data
                compare_avg_vals = avg_vals
            else:
                # compare trace data to first scenario and store plot data
                compared_trace = compare_traces(avg_vals, compare_avg_vals, comparison)

                compared_plot_data += [dict(
                    x=timesteps,
                    y=compared_trace,
                    label=scenario_labels[scenario_index-1],
                    levels=levels,
                    linewidth=linewidth,
                    color=colors[scenario_index-1] if colors is not None else None,
                    linestyle=linestyles[scenario_index-1] if linestyles is not None else None)]

        # generate plots
        sns.set_style(style)
        fig, ax = plt.subplots(figsize=fig_size)

        # plot each trace
        [single_plot(d) for d in compared_plot_data]

        # using x range of [0, steps-1] to plot initial value at 0
        ax.set_xlim([0, len(compared_plot_data[0]['x'])-1])

        set_y_limits(ax, y_limits, levels, normalize_levels, comparison)

        # set xticks and labels if provided
        if timesteps_list != []:
            plt.xticks(timesteps_list)
            ax.set_xlim([timesteps_list[0], timesteps_list[-1]])
        if timesteps_labels != [[],['']]:
            plt.xticks(timesteps_labels[0], timesteps_labels[1], rotation='vertical')

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(element)

        # add legend
        handles, labels = ax.get_legend_handles_labels()
        plt.legend(handles, labels, frameon=False)

        plt.tight_layout()
        fig.tight_layout()

        comparison_plots[element] = fig

        plt.close()

    return comparison_plots


def plot_processed_average(
        trace_data_list: list(),
        elements_list=[''],
        normalize_levels=True,
        scenario_labels=[''],
        processing='log',
        log_base=10,
        timesteps_list=[],
        timesteps_labels=[[],['']],
        y_limits=['auto'],
        x_label='Step',
        y_label='Processed Level',
        style='whitegrid',
        linewidth=1,
        colors=None,
        linestyles=None,
        fig_size=(5,3)):
        """Plot processed average traces from frequency summary in simulation results

        Inputs:
            processing: ['log', 'foldchange']
                log - plot log of trace data, use log_base as the base
                foldchange - plot fold change from first time step for each trace
        """

        processing_opts = ['log', 'foldchange']
        processing_opts_str = ', '.join([opt for opt in processing_opts])
        if processing not in processing_opts:
            raise ValueError(f'Invalid processing input, options are: {processing_opts_str}')

        if scenario_labels == ['']:
            scenario_labels = [f'Scenario {idx}' for idx, _ in enumerate(trace_data_list)]

        # check length of inputs
        if len(trace_data_list) != len(scenario_labels):
            raise ValueError('Length of scenario_labels must equal length of trace_data_list')

        if colors is not None and len(trace_data_list) != len(colors):
            raise ValueError('Length of colors must equal length of trace_data_list')

        if linestyles is not None and len(trace_data_list) != len(linestyles):
            raise ValueError('Length of linestyles must equal length of trace_data_list')

         # will get traces from each element across scenarios
        processed_plots = defaultdict()

        if elements_list == ['']:
            # assuming same elements in each scenario
            elements_list = trace_data_list[0].keys()

        for element in elements_list:
            processed_plot_data = []
            all_element_data = [trace_data[element] for trace_data in trace_data_list]

            for scenario_index, scenario_data in enumerate(all_element_data):

                levels = int(scenario_data['levels'])

                if normalize_levels:
                    # average level as percentage
                    avg_vals = scenario_data['avg_percent']
                else:
                    avg_vals = scenario_data['avg']

                # get only the specified time steps
                timesteps = list(range(len(avg_vals)))
                if timesteps_list != []:
                    timesteps = [timesteps[int(step)] for step in timesteps_list]
                    avg_vals = [avg_vals[int(step)] for step in timesteps_list]

                processed_trace = process_trace(avg_vals, processing, log_base)

                processed_plot_data += [dict(
                        x=timesteps,
                        y=processed_trace,
                        label=scenario_labels[scenario_index],
                        levels=levels,
                        linewidth=linewidth,
                        color=colors[scenario_index] if colors is not None else None,
                        linestyle=linestyles[scenario_index] if linestyles is not None else None)]

            # generate plots
            sns.set_style(style)
            fig, ax = plt.subplots(figsize=fig_size)

            # plot each trace
            [single_plot(d) for d in processed_plot_data]

            # using x range of [0, steps-1] to plot initial value at 0
            ax.set_xlim([0, len(processed_plot_data[0]['x'])-1])

            set_y_limits(ax, y_limits, levels, normalize_levels)

            # set xticks and labels if provided
            if timesteps_list != []:
                plt.xticks(timesteps_list)
                ax.set_xlim([timesteps_list[0], timesteps_list[-1]])
            if timesteps_labels != [[],['']]:
                plt.xticks(timesteps_labels[0], timesteps_labels[1], rotation='vertical')

            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.title(element)

            # add legend if more than 1 scenario
            if len(trace_data_list) > 1:
                handles, labels = ax.get_legend_handles_labels()
                plt.legend(handles, labels, frameon=False)

            plt.tight_layout()
            fig.tight_layout()

            processed_plots[element] = fig

            plt.close()

        return processed_plots


def compare_traces(this_scenario_trace, compare_scenario_trace, comparison):

    # FIXME: using 0.001 instead of 0 values to avoid div by 0 errors
    zero_val = 0.001

    if comparison == 'difference':
        # compare values at each point by subtraction of
        # the scenario 0 value at that point
        compared_trace = list([yn-y0 for yn,y0 in zip(this_scenario_trace, compare_scenario_trace)])

    elif comparison == 'percentchange':
        # compare values at each point by calculating percent change from
        # the scenario 0 value at that point
        # avoiding division by zero by using zero_val
        compared_trace = list([100*(yn-y0)/y0 if y0!=0 else (yn-zero_val)/zero_val
                for yn,y0 in zip(this_scenario_trace, compare_scenario_trace)])

    elif comparison == 'division':
        # compare values at each point by dividing by
        # the scenario 0 value at that point
        # avoiding division by zero by using zero_val
        compared_trace = list([yn/y0 if y0!=0 else yn/zero_val
                for yn,y0 in zip(this_scenario_trace, compare_scenario_trace)])

    return compared_trace


def process_trace(this_trace, processing, log_base):

    # TODO: use numpy arrays for all transformations

    # FIXME: using 0.001 instead of 0 values to avoid div by 0 errors
    zero_val = 0.001

    if processing == 'foldchange':
        # calculate fold change as the value at each step
        # divided by the value at the first step (starting value)
        y0 = this_trace['y'][0]
        processed_trace = list([yn/y0 if y0!=0 else yn/zero_val for yn in this_trace])

    elif processing == 'log':
        processed_trace = list([math.log(yn,log_base) if yn!=0 else math.log(zero_val,log_base) for yn in this_trace])

    return processed_trace


def single_plot(d, errorbars=False):
    plt.plot(d['x'], d['y'], label=d['label'], linewidth=d['linewidth'], color=d['color'], linestyle=d['linestyle'])
    if errorbars:
        var_low = np.array(d['y']) - np.array(d['error'])
        var_high = np.array(d['y']) + np.array(d['error'])
        plt.fill_between(d['x'], var_low, var_high, alpha=0.5)


def set_y_limits(ax, y_limits, levels, normalize_levels=True, comparison=None):

    if normalize_levels:
        y_max = 100
    else:
        y_max = levels - 1

    if y_limits[0] == 'default':
        # assuming all data from the same element has the same levels
        # making the y range 2% wider than levels to make sure that
        # plot lines are visible at the min and max level
        if comparison == 'difference':
            ax.set_ylim([-1.01*y_max, 1.01*y_max])
        elif comparison is None:
            ax.set_ylim([-0.01*y_max, 1.01*y_max])
    elif y_limits[0] == 'auto':
        pass
    elif len(y_limits) != 2:
        raise ValueError('Invalid y_limits input, must be [<min>, <max>]')
    else:
        # use input y_limits
        ax.set_ylim([y_limits[0], y_limits[1]])


def save_plots(element_plots, output_filepath, file_format='png' ):
    """Save plots to file(s)"""

    for element, element_fig in element_plots.items():
        element_fig.savefig(os.path.join(output_filepath,f'{element}.{file_format}'), format = file_format)


def main():

    # Parse command line arguments
    parser = argparse.ArgumentParser(
            description='Visualize simulation traces: average (default), heatmap, or individual runs',
            formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('trace_files', type=str,
                        help='names of trace files or folder of trace files to use for plotting')
    parser.add_argument('output_folder', type=str,
                        help='path to use for the output figures')
    parser.add_argument('--elements', '-e', type=str, default='',
                        help='comma-separated list of element names, specifying which elements to plot')
    parser.add_argument('--heatmap', action='store_true',
                        help='plot a heatmap of individual runs')
    parser.add_argument('--plot_each_run', action='store_true',
                        help='plot each run as its own line on the same plot axes')
    parser.add_argument('--normalize', action='store_true',
                        help='plot as percentage level')
    parser.add_argument('--errorbars', action='store_true',
                        help='plot error bars (average plots only)')
    parser.add_argument('--labels', type=str, default='',
                        help='comma-separated list of scenario labels for legends (average plots only)')
    parser.add_argument('--difference', action='store_true',
                        help='plot difference compared to first scenario (average plots only)')
    parser.add_argument('--format', choices=['png', 'pdf'], default='png',
                        help='plot output file format')

    args = parser.parse_args()

    if len(args.elements)>0 and args.elements[0]=='[':
        args.elements = args.elements[1:-1]

    elements_list = args.elements.split(',')
    labels_list = args.labels.split(',')

    if os.path.isdir(args.trace_files):
        trace_files_list = sorted(glob.glob(args.trace_files+'*.txt'))
    else:
        trace_files_list = args.trace_files.split(',')

    if args.heatmap:
        if len(trace_files_list) > 1:
            raise ValueError('Heatmap not supported for more than one input file')
        trace_data = get_traces(trace_files_list[0])
        heatmap_plots = plot_heatmap(trace_data, elements_list, args.normalize)
        save_plots(heatmap_plots, args.output_folder, args.format)

    elif args.plot_each_run:
        if len(trace_files_list) > 1:
            raise ValueError('Plot each run not supported for more than one input file')
        trace_data = get_traces(trace_files_list[0])
        each_run_plots = plot_each_run(trace_data, elements_list, args.normalize)
        save_plots(each_run_plots, args.output_folder)

    elif args.difference:
        trace_data_list = [get_traces(trace_file) for trace_file in trace_files_list]
        difference_plots = plot_compared_average(trace_data_list, elements_list, args.normalize, labels_list)
        save_plots(difference_plots, args.output_folder)

    else:
        trace_data_list = [get_traces(trace_file) for trace_file in trace_files_list]
        average_plots = plot_average(trace_data_list, elements_list, args.normalize, labels_list, args.errorbars)
        save_plots(average_plots, args.output_folder)


if __name__ == '__main__':
    main()
