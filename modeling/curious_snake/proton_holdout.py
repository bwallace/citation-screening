################################################# 
#
# run hold out experiments on proton beam
#
##################################################

import os
# was ".., old, s"
feature_sets = [os.path.join("data", "proton_beam",s) for s in ["proton_combined"]]#["proton_titles", "proton_abstracts", "proton_keywords"]]
import curious_snake
# todo: make learner setup function parameteric so you can pass it in here
curious_snake.run_experiments_hold_out(feature_sets, os.path.join("output", "proton_cofl_ODDS_RATIO_uncertainty"), num_runs=10, upto=1000, hold_out_p=.25)

