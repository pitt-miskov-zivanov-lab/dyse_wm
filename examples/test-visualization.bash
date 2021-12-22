FRAMEWORKPATH='..'

echo -e "\n\nTesting plotting average traces, multiple scenarios\n"
# plot average traces, multiple scenarios for specific elements, with errorbars
python ${FRAMEWORKPATH}/Visualization/visualization_interface.py \
traces/example_traces_PCC_0.txt,traces/example_traces_PCC_1.txt,traces/example_traces_PCC_2.txt \
plots/ \
-e AKTpf_cytoPCC,MEK1pf_cytoPCC \
--errorbars

echo -e "\n\nTesting heatmap\n"
# plot a heatmap of individual runs, normalize level as percentage
python ${FRAMEWORKPATH}/Visualization/visualization_interface.py \
traces/example_traces_all_PCC.txt \
plots/heatmaps/ \
-e AKTpf_cytoPCC \
--heatmap \
--normalize

echo -e "\n\nTesting plotting difference between scenarios\n"
# plot difference between scenarios 
python ${FRAMEWORKPATH}/Visualization/visualization_interface.py \
traces/example_traces_PCC_0.txt,traces/example_traces_PCC_1.txt,traces/example_traces_PCC_2.txt \
plots/difference/ \
-e AKTpf_cytoPCC,MEK1pf_cytoPCC \
--difference

# # TODO: add checking, optimization, sensitivity visualization examples (need CLI in viz modules first)
# echo -e "\n\nTesting plotting model checking estimates\n"
# # plot model checking estimates 

# echo -e "\n\nTesting plotting optimization results\n"
# # plot steady state optimization analysis results

# echo -e "\n\nTesting plotting sensitivity analysis results\n"
# # plot sensitivity analysis results





