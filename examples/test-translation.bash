
FRAMEWORKPATH='..'

echo -e "\n\nTesting interaction formatting and processing\n"
# Process classified interactions, output schema format
python ${FRAMEWORKPATH}/Translation/interactions.py \
interactions/example_ic_output.csv \
interactions/example_interactions.xlsx

echo -e "\n\nTesting interaction formatting and processing, ic output format\n"
# Process classified interactions, keep IC format
python ${FRAMEWORKPATH}/Translation/interactions.py \
interactions/example_ic_output.csv \
interactions/example_ic_output_processed.csv \
-o ic

echo -e "\n\nTesting interaction comparison\n"
# compare interaction sets
python ${FRAMEWORKPATH}/Translation/interactions.py \
interactions/example_ic_output.csv,interactions/example_ic_output.csv  \
interactions/example_ic_output_compared.xlsx \
--compare \
-o schema

echo -e "\n\nTesting model processing\n"
# Process model file 
MODELNAME='PCC'
python ${FRAMEWORKPATH}/Translation/model.py \
models/example_model_${MODELNAME}.xlsx \
models/example_model_${MODELNAME}_processed.xlsx

echo -e "\n\nTesting conversion of model to edges\n"
# convert model to edges for cytoscape
python ${FRAMEWORKPATH}/Translation/model.py \
models/example_model_PCC.xlsx \
graphs/example_model_edges_PCC.xlsx \
-o edges
