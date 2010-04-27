import os
#feature_sets = [os.path.join("data", "copd", s) for s in ["copd_combined"]]
feature_sets = [os.path.join("data", "copd", s) for s in ["copd_labeled_terms_only"]]
import curious_snake
# todo: make learner setup function parameteric so you can pass it in here
init_ids = eval(open(os.path.join("data", "copd", "init_ids"), 'r').readline())
curious_snake.run_experiments_finite_pool(feature_sets, os.path.join("output", "copd_labeled_features_AL"), num_runs=10, hold_out_p=.25,\
                list_of_init_ids=init_ids)