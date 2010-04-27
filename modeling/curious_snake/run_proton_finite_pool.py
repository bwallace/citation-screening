################################################# 
#
# run finite experiments on proton beam
#
##################################################

import os
#feature_sets = [os.path.join("data", "proton_beam",s) for s in ["proton_titles", "proton_abstracts", "proton_keywords"]]
#feature_sets = [os.path.join("data", "proton_beam", "old",s) for s in ["proton_titles", "proton_abstracts", "proton_keywords"]]

#feature_sets = [os.path.join("data", "proton_beam",s) for s in ["proton_combined"]]
feature_sets = [os.path.join("data", "proton_beam",s) for s in ["proton_labeled_features"]]
init_ids = eval(open(os.path.join("data", "proton_beam", "init_ids"), 'r').readline())
import curious_snake
# todo: make learner setup function parameteric so you can pass it in here

curious_snake.run_experiments_finite_pool(feature_sets, os.path.join("output", "FAKER_labeled_features_proton_dux"),\
                                                                        list_of_init_ids=init_ids, num_runs=10, upto=1000)
