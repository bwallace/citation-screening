################################################# 
#
# run experiments on trec100 data
#
# sample usage (from ipython console)
# >> %run micro_finite_pool
#
##################################################

import os
feature_sets = [os.path.join("data", "micro_nutrients",s) for s in ["titles", "abstracts", "keywords", "title_concepts", "topic_dists"]]
import curious_snake
# todo: make learner setup function parameteric so you can pass it in here
curious_snake.run_experiments_finite_pool(feature_sets, os.path.join("output", "micro_finite"), num_runs=1)