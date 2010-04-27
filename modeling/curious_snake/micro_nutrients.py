################################################# 
#
# run experiments on micro-nutrients data
#
# sample usage (from ipython console)
# >> %run trec100.py
#
##################################################

import os
feature_sets = [os.path.join("data", "micro_nutrients",s) for s in ["titles", "abstracts", "keywords", "title_concepts", "topic_dists"]]
print "running micro nutrients experiment with datasets: %s" % feature_sets
import curious_snake
curious_snake.run_passive_mv_experiments(feature_sets, os.path.join("output", "micro"))
