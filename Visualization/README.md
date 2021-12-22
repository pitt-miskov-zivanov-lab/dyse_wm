# Visualization

## Description of files

- `visualization_interface.py` 
    - plot simulation output traces
    - specify elements, plot subplots
    - create interactive plots with basic output transformations (difference between traces, normalizing, etc)

## Usage

- `visualization_interface.py`
~~~
usage: visualization_interface.py [-h] [--elements ELEMENTS]
                                  [--timesteps TIMESTEPS]
                                  [--timesteps_labels TIMESTEPS_LABELS]
                                  [--interactive] [--plot_difference]
                                  [--normalize] [--percent_change] [--log LOG]
                                  [--fold_change] [--y_limits Y_LIMITS]
                                  [--labels LABELS] [--subplots]
                                  [--max_states MAX_STATES]
                                  traces output_folder

Visualize simulation traces.

positional arguments:
  traces                names of trace files or folder of trace files to use for plotting
  output_folder         path to use for the output figures

optional arguments:
  -h, --help            show this help message and exit
  --elements ELEMENTS, -e ELEMENTS
                        comma-separated list of element names, specifying which elements to plot
  --timesteps TIMESTEPS, -ts TIMESTEPS
                        comma-separated list of x-axis points to plot values
  --timesteps_labels TIMESTEPS_LABELS, -tl TIMESTEPS_LABELS
                        comma-separated list of x-axis points to label traces with y-axis values
  --interactive, -i     create interactive plots
  --plot_difference, -d
                        plot the difference from the tracesin the first input file (trace_in_other_file - trace_in_first_file) for each element
  --normalize, -n       normalize by dividing by values of traces in the first input file
  --percent_change, -pc
                        plot percent change from traces in the first input file
  --log LOG             calculate logN(Y) for each point, where N is the input --log=N
  --fold_change, -fc    plot fold change, calculated as the ratio of the value at each step to the initial value (yn/y0)
  --y_limits Y_LIMITS, -y Y_LIMITS
                        specify the y-axis limits for plotting.
                        default: use default scale of 0,100 for activity level 
                        	or -100,100 if plotting the difference 
                        	or auto-scale for normalize or fold change 
                        auto: use auto scaling 
                        <min>,<max>: use the first value as the min, second value as the max
  --labels LABELS, -lbl LABELS
                        comma-separated list of labels to use for traces, if plottingmultiple traces on the same plots
  --subplots, -s        plot interactive figures in sets of 6 subplots
  --max_states MAX_STATES
                        maximum number of states (levels) for each variable (2 for Boolean)
~~~

## Examples

- see [`examples/test-visualization.bash`](examples/test-visualization.bash)
