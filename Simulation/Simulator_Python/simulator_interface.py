import os.path
import sys
import re
import time
import argparse
import math
from io import StringIO
import openpyxl
import pandas as pd
import numpy as np
import logging
from Simulation.Simulator_Python.simulator import Simulator
from Visualization import visualization_interface as viz

def setup_and_create_rules(inputFilename, outputFilename, scenarioString='0'):
    """Create a model object, derive update rules, output to file(s).
    """

    scenarios = scenarioString.split(',')

    model = Simulator(inputFilename) 

    # output rules for each scenario
    if len(scenarios) > 1:
        # append scenario index to file names
        for this_scenario in scenarios:
            logging.info(('Scenario: %s') % str(this_scenario))
            this_output_filename = os.path.splitext(outputFilename)[0] + '_' + str(this_scenario) + '.txt'
            model.create_rules(this_output_filename,int(this_scenario))
    else:
        logging.info(('Scenario: %s') % str(scenarios[0]))
        this_output_filename = os.path.splitext(outputFilename)[0] + '.txt'
        model.create_rules(this_output_filename,int(scenarios[0]))

    
def setup_and_create_truth_tables(inputFilename, outputBaseFilename, scenarioString='0'):
    """Create a model object, derive truth tables, output to files.
    """

    scenarios = scenarioString.split(',')

    model = Simulator(inputFilename) 

    # output truth tables for each scenario
    if len(scenarios) > 1:
        # append scenario index to file names
        for this_scenario in scenarios:
            logging.info(('Scenario: %s') % str(this_scenario))
            this_base_filename = outputBaseFilename + '_' + str(this_scenario)
            model.create_truth_tables(this_base_filename,int(this_scenario))
    else:
        logging.info(('Scenario: %s') % str(scenarios[0]))
        model.create_truth_tables(outputBaseFilename,int(scenarios[0]))


def calculate_simulation_steps(inputFilename, time_units, scale_factor=1):
    """Calculate simulation steps from time units scaled to model size
    """

    model = Simulator(inputFilename)
    elements = model.get_elements()
    num_elements = len(elements)
    num_elements -= sum([int(elements[element]._Element__act=='')*int(elements[element]._Element__inh=='')
        for element in elements])

    steps = int(np.ceil(time_units * num_elements * scale_factor))

    return steps


def calculate_toggle_steps(inputFilename, toggle_times: list(), scale_factor=1):
    """Calculate toggle steps from toggle times scaled to model size.
        
        Use with calculate_simulation_steps to scale simulation length.        
    """

    model = Simulator(inputFilename)
    elements = model.get_elements()
    num_elements = len(elements)
    num_elements -= sum([int(elements[element]._Element__act=='')*int(elements[element]._Element__inh=='')
        for element in elements])

    toggle_steps = [int(np.ceil(this_toggle * num_elements * scale_factor))
            for this_toggle in toggle_times]

    return toggle_steps


def setup_and_run_simulation(inputFilename, outputFilename, steps=1000,
        runs=100, simScheme='ra', outputFormat=3, scenarioString='0',
        normalize=False, randomizeEachRun=False, event_traces_file=None, progressReport=False):
    """Create a model object, run simulations, output traces to file(s). 
    """

    scenarios = scenarioString.split(',')
    model = Simulator(inputFilename)

    if len(scenarios) > 1:
        # append the scenario index to the end of each file name
        for this_scenario in scenarios:
            this_output_filename = os.path.splitext(outputFilename)[0] + '_' + str(this_scenario) + '.txt'
            if event_traces_file is not None:
                this_event_traces_file = os.path.splitext(event_traces_file)[0] + '_' + str(this_scenario) + '.txt'
            else:
                this_event_traces_file = None
            model.run_simulation(simScheme, runs, steps, this_output_filename, 
                int(this_scenario), outMode=outputFormat, normalize=normalize, 
                randomizeEachRun=randomizeEachRun, eventTraces=this_event_traces_file,
                progressReport=progressReport)
            logging.info(('Simulation scenario %s complete') % str(this_scenario))
    else:
        this_output_filename = os.path.splitext(outputFilename)[0] + '.txt'
        if event_traces_file is not None:
            this_event_traces_file = os.path.splitext(event_traces_file)[0] + '.txt'
        else:
            this_event_traces_file = None
        model.run_simulation(simScheme, runs, steps, this_output_filename, 
            int(scenarios[0]), outMode=outputFormat, normalize=normalize,
            randomizeEachRun=randomizeEachRun, eventTraces=this_event_traces_file,
            progressReport=progressReport)
        logging.info(('Simulation scenario %s complete') % str(scenarios[0]))


def calculate_trace_difference(inputFilenames: list,outputFilename):  
    """Calculate trace-by-trace difference between two simulation trace files.
    """

    if len(inputFilenames) == 1:
        raise ValueError('Cannot calculate difference with only one input file')   
    
    # read first input file
    first_file = inputFilenames[0]
    with open(first_file) as first:
        first_trace_data = first.readlines()
    first_trace_data = [x.strip() for x in first_trace_data]

    transpose_format = False
    # check for transposed format
    if first_trace_data[0][0] == '#':
        transpose_format = True

    # loop through other files and calculate the difference
    for file_index,this_file in enumerate(inputFilenames[1:]):
        # use scenario index at the end this_file name if file name ends with the index
        scenario_index = re.findall(r'([0-9]+).txt',this_file)
        if scenario_index:
            output_index = scenario_index[0]
        else:
            output_index = file_index
        
        output_file = open(os.path.splitext(outputFilename)[0] + str(output_index) + '.txt','w')

        with open(this_file) as this_:
            this_trace_data = this_.readlines()
        this_trace_data = [x.strip() for x in this_trace_data]

        for content_index,line in enumerate(this_trace_data):
            # check each line for the Frequency Summary
            if ('Frequency' in line) \
                or ('Run' in line) :

                out_line = line + '\n'
            elif '#' in line:

                if line != first_trace_data[content_index]:
                    raise ValueError('Element names in trace files do not match')

                out_line = line + '\n'

            else:
                this_trace_data_points = line.split(' ')
                first_trace_data_points = first_trace_data[content_index].split(' ')

                if transpose_format:
                    temp_trace_data_points = [this_trace_data_points[0]] + [this_trace_data_points[1]] \
                        + [str(int(x2) - int(x1)) for (x2,x1) in zip(this_trace_data_points[2:-1],first_trace_data_points[2:-1])] \
                        + [this_trace_data_points[-1]]
                else:
                    if (this_trace_data_points[0] != first_trace_data_points[0]):
                        raise ValueError('Element indices in trace files do not match '+str(this_trace_data_points[0]) + '\t' + str(first_trace_data_points[0]))
                    
                    temp_trace_data_points = [this_trace_data_points[0]] \
                        + [str(int(x2) - int(x1)) for (x2,x1) in zip(this_trace_data_points[1:],first_trace_data_points[1:])]

                out_line = ' '.join(temp_trace_data_points) + '\n'

            output_file.write(out_line)


def concatenate_traces(inputFilenames: list, outputFilename):
    """Concatenate simulation traces from multiple scenarios.
    """

    if len(inputFilenames) == 1:
        raise ValueError('Cannot concatenate with only one input file')

    # read all input files
    all_traces = []

    for file_index, this_file in enumerate(inputFilenames):

        traces = pd.read_table(this_file, delimiter=' ')
        if '#' not in traces.columns:
            raise ValueError('Invalid trace file format')

        # use scenario index at the end this_file name if file name ends with the index
        scenario_index = re.findall(r'([0-9]+).txt', this_file)
        if scenario_index:
            output_index = scenario_index[0]
        else:
            output_index = file_index

        traces = traces.set_index(['#', 'time'])

        # save steps column to append to the end later
        if file_index == 0:
            steps = traces[['step']]

        traces.drop(columns='step', inplace=True)

        traces.columns = [col + '_' + str(output_index) for col in traces.columns]

        all_traces += [traces]

    # include steps column, then concatenate all traces
    all_traces += [steps]
    all_traces_concat = pd.concat(all_traces, axis=1, join_axes=[all_traces[0].index])

    all_traces_concat.to_csv(outputFilename, sep=' ')

def trace_distributions(model_file:str, traces_file:str, time_units:int, scale_factor:int=1, normalize:bool=False):
    """This function returns trace distributions at designated time steps."""

    traces = viz.get_traces(traces_file)
    distributions = {k:dict() for k in traces}
    for i,t in enumerate(calculate_toggle_steps(model_file, toggle_times=range(time_units+1), scale_factor=1)):
        for el in traces:
            M = np.concatenate([np.array(v).reshape(-1,1) \
                                for v in traces[el]['traces'].values()], axis=1)
            distributions[el][i] = M[t,:]
            if normalize:
                distributions[el][i] /= (traces[el]['levels']-1)

    return distributions


def get_input_args():

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Run a simulation',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('input_filename', type=str,
                        help='path and name of the input file to use for the simulation')
    parser.add_argument('output_filename', type=str,
                        help='path and name to use for the output file')
    parser.add_argument('--sim_scheme', type=str, choices=['ra', 'round', 'sync', 'ra_multi', 'sync_multi', 'rand_sync', 'rand_sync_gauss', 'fixed_updates'], default='ra',
                        help='simulation scheme: \n'
                        'ra: random asynchronous, randomly selects and updates one element each time step. \n'
                        'round: randomly selects an element to be updated. All elements must be updated before \n'
                        'the same element can be updated again (must complete round). can use ranks to specify \n'
                        'elements which must be updated before others.\n'
                        'sync: synchronous, updates all elements at each time step.\n'
                        'ra_multi: random asynchronous multi-step, randomly select an element and select if it is updated based on probability. \n'
                        'sync_multi: synchronous multi-step, iterate through all elements, decide whether next state will hold current value \n'
                        'or compute new value, and then update all elements simultaneously \n'
                        '\t (NOTE: to get different end values with sync scheme, at least some \n'
                        '\t initial values must be randomized by specifying \'r\' in the input file)'
                        'rand_sync: synchronous with uniform random delays \n'
                        'rand_sync_gauss: synchronous with Gaussian random delays \n'
                        'fixed_updates: read order from event_traces file')
    parser.add_argument('--runs', type=int, default=100,
                        help='number of simulation runs (repetitions)')
    parser.add_argument('--steps', type=int, default=1000,
                        help='number of simulation steps (duration)')
    parser.add_argument('--output_format', '-o', type=int, default=0,
                        help='format of trace file output: \n'\
            '1 (Default for sync scheme): traces for all runs and frequency summary \n' \
            '2: traces for all runs in transpose format (for model checking or sensitivity analysis) \n' \
            '3 (Default for ra scheme): trace file with frequency summary only \n' \
            '4: model rules and truth tables \n' \
            '5: model rules \n' \
            '6: truth tables \n'
            # TODO: make event trace output a separate output option, or output both values and elements 
            '7: event traces including elements updated at each step')
    parser.add_argument('--event_traces_file', type=str, default=None,
                        help='path and name to use for the event traces file') 
    parser.add_argument('--normalize_output', '-n', action='store_true',
                        help='in traces file, normalize values to the range [0,1]')
    parser.add_argument('--scenarios',type=str, default='0', 
                        help='scenario index specifying which Initial column to use \n'\
                        'or comma-sparated list of scenarios indices \n'\
                        '(0 specifies the first Initial column)')
    parser.add_argument('--difference','-d',action='store_true',
                        help='calculate the difference between traces for the first scenario \n'\
                        'and traces from any other scenarios')
    parser.add_argument('--concatenate','-c',action='store_true',
                        help='concatenate trace files for all scenarios')
    parser.add_argument('--randomize_each_run', '-r', action='store_true',
                        help='For sequential schemes, randomize initial values in each run.')
    parser.add_argument('--timed','-t', action='store_true',
                        help='log the total simulation time')
                        
    args = parser.parse_args()

    #TODO: check all argument formats 

    if args.concatenate and args.output_format != 2:
        raise ValueError('Concatenation only supported for output_format=2')

    if args.difference and len(args.scenarios.split(',')) < 2:
        raise ValueError('Need at least two scenarios to calculate the difference')

    if args.concatenate and len(args.scenarios.split(',')) < 2:
        raise ValueError('Need at least two scenarios to concatenate trace files')

    if args.sim_scheme == 'fixed_updates' and args.event_traces_file is None:
        raise ValueError('Need to specify the event_traces file')

    return args


def main():

    args = get_input_args()

    # set up logging to stdout
    log_formatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    log_handler = logging.StreamHandler(sys.stdout)
    log_handler.setFormatter(log_formatter)
    logger = logging.getLogger()
    logger.addHandler(log_handler)
    logger.setLevel(logging.DEBUG)

    # set default trace file format
    if args.output_format == 0:
        if args.sim_scheme =='ra' or args.sim_scheme =='round' or args.sim_scheme =='ra_multi':
            output_format = 3
        elif args.sim_scheme == 'sync' or args.sim_scheme == 'sync_multi' or args.sim_scheme == 'rand_sync' or args.sim_scheme == 'rand_sync_gauss' or args.sim_scheme == 'fixed_updates':
            output_format = 1
        else:
            raise ValueError(
                'Invalid simulation scheme, choose: ra, round, sync, ra_multi, sync_multi, fixed_updates, or rand_sync')
    else:
        output_format = args.output_format

    if output_format == 3 and (args.sim_scheme == 'sync' or args.sim_scheme == 'sync_multi' or args.sim_scheme == 'rand_sync' or args.sim_scheme == 'rand_sync_guass'):
        raise ValueError('Frequency summary (output format 3) is not supported for synchronous scheme')

    if output_format in [1, 2, 3, 7]:
        # run simulation to get output traces
        logging.info('Simulating...')
        start_time = time.time()
    
        setup_and_run_simulation(args.input_filename, args.output_filename,
            args.steps, args.runs, args.sim_scheme, output_format, args.scenarios,
            args.normalize_output, args.randomize_each_run, args.event_traces_file)
        
        if args.timed:
            logging.info(('--- %0.3f seconds ---') % (time.time() - start_time))

        if args.difference:
            trace_file_names = [
                os.path.splitext(args.output_filename)[0] 
                + '_' + x + '.txt' for x in args.scenarios.split(',')
                ]

            calculate_trace_difference(trace_file_names, 
                os.path.splitext(args.output_filename)[0] + '_diff.txt')

        if args.concatenate:
            if output_format != 2:
                raise ValueError('Concatenation only supported for output format 2')
            else:
                if args.difference:
                    # skip the first scenario index when getting difference file names
                    trace_file_names = [
                        os.path.splitext(args.output_filename)[0] 
                        + '_diff' + x + '.txt' for x in args.scenarios.split(',')[1:]
                        ]
                else:
                    trace_file_names = [
                        os.path.splitext(args.output_filename)[0] 
                        + '_' + x + '.txt' for x in args.scenarios.split(',')
                        ]

                concatenate_traces(trace_file_names, 
                    os.path.splitext(args.output_filename)[0] + '_concat.txt')

    elif output_format in [4, 5, 6]:
        # get model rules and/or truth tables
        if output_format != 5:
            # get truth tables
            # TODO: mkdir for truth tables, check file path for isdir
            setup_and_create_truth_tables(args.input_filename, 
                os.path.splitext(args.output_filename)[0], args.scenarios)
        
        if output_format != 6:
            # get model rules
            setup_and_create_rules(args.input_filename, args.output_filename, args.scenarios)

    # TODO: support rule file input
    # TODO: call Java/C simulator for rule file input for now
    
    else:
        raise ValueError('Unrecognized output_format, use output_format of 1, 2, 3, 5')


if __name__ == '__main__':
    main()

