
# Simulation

Setup and usage of simulator code.
        
## Python simulator

### Description of files
- `simulator_interface.py` functions for running a simulation for a specified number of steps and runs, or creating executable model rules
        - use output format option to specify output of model rules or simulation traces, and format of simulation trace file
- `simulator.py` defines a Simulator class for models, and an Element class for model elements (nodes)

### Usage
- run `python simulator_interface.py -h`

### Examples
- for shell interface examples, see [`examples/test-simulation.bash`](examples/test-simulation.bash)
- for scripting examples, see [`examples/examples.ipynb`](examples/examples.ipynb)
