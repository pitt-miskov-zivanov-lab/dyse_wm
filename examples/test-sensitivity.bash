# # This bash contains all procedures to run sensitivity analysis, starting from a model spreadsheet

FRAMEWORKPATH='..'

MODELNAME='Tcell_boolean'
INQUIRY_ELEMENT='PTEN'
INQUIRY_PATHWAY='TCR_HIGH,FOXP3&IL2,sensi,2'

# # Scenarios definition is required to run dynamic analysis, where initial states of model matter
# # Choose the scenario(s) you'd like to study and analyze, seperated by comma, e.g.,SCENARIOS=0 or SCENARIOS=0,1,2
SCENARIOS=0,1

# # To run dynamic analysis, trace files are needed. With this bash script, you don't need to run test-simulator.bash to get them
# # Chosse your simulation parameters, see README.md of simulator.py for details
STEPS=100
RUNS=100

# # This is a part of simulator_interface.py: Convert model spreadsheet into rules output in .txt type
python ${FRAMEWORKPATH}/Simulation/Simulator_Python/simulator_interface.py \
models/example_model_${MODELNAME}.xlsx \
rules/example_rules_${MODELNAME}.txt \
--output_format=5 \
--scenarios=${SCENARIOS}

# # Static Analysis
# parse model rules (.txt file) to obtain static sensitivity analysis raw results
echo "Parse $MODELNAME model, static:"
${FRAMEWORKPATH}/Sensitivity/Static/analysis \
rules/example_rules_${MODELNAME}.txt \
> sensitivity/example_static_analysis_${MODELNAME}

# obtain more detailed static sensitivity analysis results via option{-a} or inquired results from users via options{-e} or {-p}
# refer to python sensitivity.py -h for instructions
python ${FRAMEWORKPATH}/Sensitivity/Pathway_Analysis/sensitivity.py \
-l 'static' \
-i sensitivity/example_static_analysis_${MODELNAME} \
-o sensitivity/${MODELNAME} \
-a 'ALL' \
-e $INQUIRY_ELEMENT \
-p $INQUIRY_PATHWAY

# # This is a part of simulator_interface.py: Obtain transpose trace file output (used by dynamic sensitivity analysis)
# # Note: if there is only one scenario you'd like to study and run to get traces, you shall obtain example_traces_transpose_${MODELNAME}.txt
# # if there are multiple scenarios, you should obtain multiple traces, named as example_traces_transpose_${MODELNAME}_index.txt
python ${FRAMEWORKPATH}/Simulation/Simulator_Python/simulator_interface.py \
models/example_model_${MODELNAME}.xlsx \
sensitivity/example_traces_transpose_${MODELNAME}.txt \
--sim_scheme=ra \
--output_format=2 \
--steps=${STEPS} \
--runs=${RUNS} \
--scenarios=${SCENARIOS}

# # In order to handle the issue of name indexing when running multiple scenarios and differentiate them. The following for-loop and if-else commands are used
IFS=',' read -r -a array <<< "$SCENARIOS"
for i in "${array[@]}"
do
  if [ ${#array[@]} -eq 1 ];
  then
    MODELNAME_INDEX="${MODELNAME}"
  else
    MODELNAME_INDEX="${MODELNAME}_$i"
  fi

  printf "\n"
  echo "Parse $MODELNAME_INDEX model, dynamic:"

  # # Dynamic Analysis
  # parse model rules (.txt file) and trace file (could be multiple, therefore for-loop is used) to obtain dynamic analysis raw results
  ${FRAMEWORKPATH}/Sensitivity/Dynamic/DyAnalysis \
  sensitivity/example_traces_transpose_${MODELNAME_INDEX}.txt \
  sensitivity/example_static_analysis_${MODELNAME} \
  sensitivity/example_dynamic_analysis_${MODELNAME_INDEX}

  # obtain more detailed dynamic sensitivity analysis results via option{-a} or inquired results from users via options{-e} or {-p}
  # refer to python sensitivity.py -h for instructions
  python ${FRAMEWORKPATH}/Sensitivity/Pathway_Analysis/sensitivity.py \
  -l 'dynamic' \
  -i sensitivity/example_dynamic_analysis_${MODELNAME_INDEX} \
  -o sensitivity/${MODELNAME_INDEX} \
  -a 'ALL' \
  -e $INQUIRY_ELEMENT \
  -p $INQUIRY_PATHWAY
done
