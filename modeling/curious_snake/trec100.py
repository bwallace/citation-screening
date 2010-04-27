################################################# 
#
# run experiments on trec100 data
#
# sample usage (from ipython console)
# >> %run trec100.py
#
##################################################

import os
feature_sets = [os.path.join("data", "trec100",s) for s in ["titles", "abstracts", "keywords", "title_concepts","abstracts_400_topics"]]
import curious_snake
curious_snake.run_passive_mv_experiments(feature_sets, os.path.join("output", "trec100"))
