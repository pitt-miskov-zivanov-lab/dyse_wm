FRAMEWORKPATH='..'

MODELNAME='PCC'
STEPS=100

# TODO: create a smaller example model and add tests for simulation schemes

echo -e "\n\nTesting simulation, frequency summary output\n"
# default trace file output (frequency summary only)
python ${FRAMEWORKPATH}/Simulation/Simulator_Python/simulator_interface.py \
models/example_model_${MODELNAME}.xlsx \
traces/example_traces_${MODELNAME}.txt \
--steps=${STEPS} 

echo -e "\n\nTesting simulation, multiple scenarios\n"
# Running multiple scenarios
python ${FRAMEWORKPATH}/Simulation/Simulator_Python/simulator_interface.py \
models/example_model_${MODELNAME}.xlsx \
traces/example_traces_${MODELNAME}.txt \
--scenarios=0,1,2 \
--steps=${STEPS}

echo -e "\n\nTesting simulation, all runs output\n"
# all simulation runs trace file output
python ${FRAMEWORKPATH}/Simulation/Simulator_Python/simulator_interface.py \
models/example_model_${MODELNAME}.xlsx \
traces/example_traces_all_${MODELNAME}.txt \
--output_format=1 \
--runs=100 \
--steps=${STEPS} \
--randomize_each_run

echo -e "\n\nTesting simulation, event trace output\n"
# event trace output
python ${FRAMEWORKPATH}/Simulation/Simulator_Python/simulator_interface.py \
models/example_model_${MODELNAME}.xlsx \
traces/example_event_traces_${MODELNAME}.txt \
--steps=${STEPS} \
--output_format=7

echo -e "\n\nTesting simulation, fixed update scheme using event traces\n"
# event trace input for fixed updates
python ${FRAMEWORKPATH}/Simulation/Simulator_Python/simulator_interface.py \
models/example_model_${MODELNAME}.xlsx \
traces/example_fixed_updates_${MODELNAME}.txt \
--steps=${STEPS} \
--sim_scheme=fixed_updates \
--event_traces_file=traces/example_event_traces_${MODELNAME}.txt 

echo -e "\n\nTesting simulation, transpose trace output\n"
# transpose trace file output (used by model checking and sensitivity)
python ${FRAMEWORKPATH}/Simulation/Simulator_Python/simulator_interface.py \
models/example_model_${MODELNAME}.xlsx \
traces/example_traces_transpose_${MODELNAME}.txt \
--output_format=2 \
--steps=${STEPS} \
--runs=1

echo -e "\n\nTesting simulation, model with weights\n"
# model with weights
# TODO: add weights to example model
MODELNAME='Tcell_weights'
python ${FRAMEWORKPATH}/Simulation/Simulator_Python/simulator_interface.py \
models/example_model_${MODELNAME}.xlsx \
traces/example_traces_${MODELNAME}.txt \
--steps=${STEPS} 

echo -e "\n\nTesting simulation, model with delays\n"
# model with delays
# TODO: add delays to example model
MODELNAME='delays'
STEPS=10
python ${FRAMEWORKPATH}/Simulation/Simulator_Python/simulator_interface.py \
models/example_model_${MODELNAME}.xlsx \
traces/example_traces_all_${MODELNAME}.txt \
--output_format=1 \
--runs=2 \
--sim_scheme='rand_sync' \
--steps=${STEPS}

echo -e "\n\nTesting simulation, rules output\n"
MODELNAME='Tcell'
# rules output
python ${FRAMEWORKPATH}/Simulation/Simulator_Python/simulator_interface.py \
models/example_model_${MODELNAME}.xlsx \
rules/example_rules_${MODELNAME}.txt \
--output_format=5 
