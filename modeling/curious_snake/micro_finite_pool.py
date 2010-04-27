################################################# 
#
# sample usage (from ipython console)
# >> %run micro_finite_pool
#
##################################################

import os
#feature_sets = [os.path.join("data", "micro_nutrients",s) for s in ["abstracts"]]
feature_sets = [os.path.join("data", "micro_nutrients",s) for s in ["micro_labeled_features_only"]]
import curious_snake
# todo: make learner setup function parameteric so you can pass it in here
curious_snake.run_experiments_finite_pool(feature_sets, os.path.join("output", "micro_abstracts_finite_pool_labeled_feature_space"), num_runs=10, upto=2000)