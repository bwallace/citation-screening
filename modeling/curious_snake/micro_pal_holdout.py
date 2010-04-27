################################################# 
#
# run hold out experiments on micro 
#
# sample usage (from ipython console)
# >> %run micro_pal_holdout.py
#
##################################################

import os
feature_sets = [os.path.join("data", "micro_nutrients",s) for s in ["micro_t_and_a"]]
import curious_snake

# todo: make learner setup function parameteric so you can pass it in here
curious_snake.run_experiments_hold_out(feature_sets, os.path.join("output", "micro_combined_ODDS_RATIO_COTESTING"), num_runs=10, upto=1000, hold_out_p=.25)
