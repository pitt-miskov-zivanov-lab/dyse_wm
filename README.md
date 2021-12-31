# DySE Framework 

## Framework description

The Dynamic System Explanation (DySE) Framework contains tools for building, simulating, and verifying qualitative models, built by the Mechanisms and Logic of Dynamics Lab at the University of Pittsburgh.

## In this README

This README contains installation instructions, description of the overall framework, and for each subpackage: functionality, and input/output files. READMEs for each subpackage contain descriptions of each subpackage module. 

## Setup

### Install DySE 

After cloning the repository, run from inside the top-level directory:

~~~shell
pip install -e .
~~~

## Package Structure

### Current structure

- [`CONTRIBUTING.md`](CONTRIBUTING.md)
    - Read [`CONTRIBUTING.md`](CONTRIBUTING.md) if you are starting development
    - Describes coding environment setup, coding guidelines, resources, etc.
- `README.md`
    - (You Are Here)
    - Describes general functionality of each module, input/output requirements
    - More detailed READMEs with usage are in each package within the framework
- `setup.py`
    - local installation instructions
- examples/
    - example usage scripts and input/output files for each module
- Translation/
- Simulation/
- Sensitivity/
- Visualization/

---
## Examples
- example bash scripting: `examples/test-*.bash`
- example Python scripting: `examples/examples.ipynb`
~~~shell
jupyter notebook examples/examples.ipynb
~~~

___
## Translation

### Functionality

- Process and translate files into formats compatible with later stages in the framework
- for usage, see [`Translation/README.md`](Translation/README.md) or [`examples/examples.ipynb`](examples/examples.ipynb) or [`examples/test-translation.bash`](examples/test-translation.bash)

### I/O

- Interactions
    - File Input: interactions from automated reading or IC output
        - `examples/interactions/example_bio_interactions.xlsx`
        - `examples/interactions/example_ic_output.csv`
    - File Output: interactions in standardized tabular format (.xls/.xlsx)
        - `examples/interactions/example_interactions.xlsx`

- Models    
    - File Input: model file in tabular format (.xls/.xlsx) 
        - `examples/example_model*.xlsx`
    - File Output:
        - processed model file in tabular format (.xls/.xlsx)
            - `examples/example_model*.xlsx`
        - model edges (for graphing in Cytoscape) (.xls/.xlsx)
            - `examples/example_model_edges*.xlsx`
    
---
## Simulation

### Functionality

- Simulate executable model for a specified number of simulation steps and runs
- for usage, see [`Simulation/README.md`](Simulation/README.md) or [`examples/examples.ipynb`](examples/examples.ipynb) or [`examples/test-simulation.bash`](examples/test-simulation.bash)

### I/O
- File Input: model file in tabular format (.xls/.xlsx) 
    - `examples/models/example_model*.xlsx`
    - Required columns:

    - | Element Name | Element IDs | Element Type | Variable | Positive Regulators | Negative Regulators | Scenario | 

        - Variables should only contain letters, numbers, and underscores
        - Variables should not start with a number
        - In Positive regulators and Negative regulators columns, use influence set notation:
            - A, B : Logical OR
                - A OR B, which is max(A,B) for discrete variables
            - (A, B) : Logical AND 
                - A AND B, which is min(A,B) for discrete variables
            - {A}[B] : Necessary pair 
                - A itself is sufficient for regulating X, while B can only strengthen the regulation when A is present. 
                - A and B can be expressions
                - The score is calculated as follows: 
                    - If weights are included within {} or [], and some element within {} is nonzero, the score is calculated by summation: sum(A) + sum(B) 
                    - If weights are not included, and some element within {} is nonzero, the score is calculated as: max(min(A),max(B)) 
                    - If all elements within {} are zero, the score is 0
            - A^ : Highest-value regulator 
                - A only has an effect if A is at its highest value
                - The score is either 0 or the maximum value
            - A=1 : Target-value regulator
                - A only has an effect if A is at the target value (e.g., A=0 means A will only have an effect if A's value is 0)
                - The score is either 0 or the maximum value
            - !A : Logical NOT
                - for discrete variables, uses n's complement
            - A + B : Summation
            - 2*A : Weight 
                - The effect of A will be the current value of A multiplied by the weight 
            - 6~A : Propagation delay
                - The effect of A on the regulated element is delayed (contrast with regulation delays, specified in the Delays column for the regulated element)
            - {A} : Initializer 
                - Suppose we are updating variable X, and A is the initializer. If X is 0, then we must have A be >0 to have X activated
            - Mixed OR and summation is not valid
            - Alternatively, include a truth table mapping the regulators and next-state value in other sheets of the model workbook, in the format below, where the first cell is the name of the regulated element, the next row contains the names of the regulators, and the subsequent rows are rows of the truth table.
                - TODO: add example model with table update function
            
                | AKT	|			

                | MTORC2	| PDK1	| AKT_OFF	| AKT	| Value |

                | 0	| 0	| 0	| 0	| 0 |

                | 0	| 0	| 1	| 0 | 0 |

                | 0	| 1	| 0	| 0	| 0 |

                ...
        - Initial simulation values
            - Can have multiple columns including the word "Scenario" and use the scenario input to simulator_interface to specify which scenario to run
            - encode a toggle with the notation below, which specifies toggling from 0 to 1 at time step 300:
                - 0,1[300]
            - randomize with input 'r' or 'random'
            - can also input 'l'/'low' 'm'/'med'/'middle' or 'h'/'high'

    - Optional columns:
        - Levels
            - number of discrete variable levels for that element (default is 3)
        - Increment Multiplier
            - specify a number greater than 0 to set the increment as proportional to the difference between positive and negative regulation scores, multiplied by the input number
                - e.g., for an input of 3, the increment will be the difference between positive and negative (in terms of levels) multiplied by 3
            - default is 1
            - if set to 0, the increment when an element is updated is always 1 or -1 level depending on whether positive or negative regulation is greater, respectively
        - Delay
            - state transition delays in the format delay01,delay12,delay21,delay10 for 3 states (e.g., 1,1,1,1)
            - if only one delay is listed it will be used for all state transitions
            - leave blank for no delays
        - Balancing
            - specifies what happens when positive and negative regulation scores are equal, with optional delay
            - "decrease,0" is the default (if the column is left blank), and specifies a decrease with 0 delay
            - input "None" for no balancing behavior
        - Spontaneous 
            - specifies spontaneous behavior for elements with either no positive or no negative regulators
            - input as an integer specifying delay in spontaneous behavior: "0" specifies spontaneous behavior with no delay
            - input "None" for no spontaneous behavior
                - WARNING! this will mean that an element has no way of returning to 0 if it has only positive regulators, for example
            - if a value is input but the element has both positive and negative regulators, there will be no spontaneous effect
        - Update Group
            - for group-based simulation schemes
            - elements in the same group will be updated in the same simulation step
        - Update Rate
            - for Random Asynchronous simulation
            - input must be an integer
            - elements with higher update rate values will be updated at a greater rate than those with lower values
            - Probability to get selected scales linearly with update rate value. Elements with update rate value of 4 are 4 times as likely to be run as elements with update rate value of 1  and twice as likely to be run as elements with update rate value of 2.
            - Elements of unspecified update rate values are assumed to have an update rate value of 1.
            - If you want a particular element to be chosen at half the rate of every other rule, you would have to make every other rule a 2. Essentially find the Least common multiple as an integer and scale based on that. i.e. if element A should have a rate of 0.33, B a rate of 1, and C a rate of 2, your column values for A should be 1 (or left blank), B should be 3, and C should be 6.
        - Update Rank
            - For round based simulation. Elements with higher update rank will be run before elements with lower update rank.
            - Input must be an integer.
            - Elements of unspecified update rank values will be assigned one of 0.
            - If you want a group to be run after everything (like a priority of -1), you will have to leave its value as 0 (or blank), and give everything else a priority of 1
        - Update Probability
            - For Random Asynchronous multi-step simulation. 
            - Input must be a float.
            - Elements with higher probability are more likely to get updated, and a random number of elements is updated in each step.
            - RNG generates a float between [0.0,1.0). If the assigned probability is greater than the generated number the check is passed and the element is updated 
            - Thus all values below and equal to 0 will never be updated, all values greater than or equal 1.0 will always be updated.
            - Default value is 0.5 
        - Optimization Input/Output
            - for elements involved in steady state optimization analysis (leave blank for other elements)
            - 'I', 'i', 'Input', 'input' to fix input value (in value column described below)
            - 'O', 'o', 'Output', 'output' to set target output value (in value column described below)
        - Optimization Fixed Input/Ouput Value
            - numerical level to use for steady state optimization (represents steady state value)
        - Optimization Objective Weight
            - weight to use in steady state optimization cost function
        - Data
            - hyphen-separated lists of values for x axis and y axis, e.g., [0, 6, 22, 50]-[0, 10, 25, 33]
            - first list is x axis values, second list is y axis
            - used for delay optimization
            - include one column for each scenario, labeled similar to scenario columns (e.g., data 0, data 1)

- File Output: simulation traces (.txt) 
    - `examples/traces/example_traces*.txt`
---
## Sensitivity

### Functionality

- Perform sensitivity analysis 
    - static : on model structure
    - dynamic : using simulation traces and model structure
- for usage, see [`Sensitivity/README.md`](Sensitivity/README.md) or [`examples/examples.ipynb`](examples/examples.ipynb) or [`examples/test-sensitivity.bash`](examples/test-sensitivity.bash)

### I/O
#### Dynamic
- File Input:
    - simulation trace in transpose format (.txt) 
        - `examples/sensitivity/example_traces*.txt`
    - static sensitivity analysis results
        - `examples/sensitivity/example_static_analysis*`
- File Output: dynamic sensitivity analysis results
    - `examples/sensitivity/example_dynamic_analysis*`

---

## Visualization

### Functionality

- Visualize plots of dynamic simulation traces
- for usage, see [`Visualization/README.md`](Visualization/README.md) or [`examples/examples.ipynb`](examples/examples.ipynb) or [`examples/test-visualization.bash`](examples/test-visualization.bash)

### I/O

- File Input: trace file output from simulation (.txt)
    - `examples/traces/example_traces*.txt`
- File Output: .pdf or .png plots
    - `examples/plots/*`

---
## History and Support
The DySE framework development team is based in MeLoDy Lab led by Dr. Natasa Miskov-Zivanov, in the Electrical and Computer Engineering Department at the University of Pittsburgh. DySE was initially developed as part of AIMCancer (Automated Integration of Mechanisms in Cancer) project and later revised and extended as part of STORM (Standardized Technology for Optimizing Rapid Modeling) project.

Since its initiation in 2014, many researchers and developers have contributed to DySE including Kara Bocan, Khaled Sayed, Stefan Andjelkovic, Gaoxiang Zhou, Yasmine Ahmed, Emilee Holtzapple, Casey Hansen, Yu-Hsin Kuo, Kai-Wen Liang, Cheryl Telmer, Natasa Miskov-Zivanov. 

DySE development has been supported by DARPA Big Mechanism (AIMCancer, W911NF-17-1-0135) and World Modelers (STORM, W911NF-18-1-0017) program grants, and by the Department of Electrical and Computer Engineering at the University of Pittsburgh Swanson School of Engineering.
