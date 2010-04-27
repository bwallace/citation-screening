import os
feature_sets = [os.path.join("data", "copd", s) for s in ["copd_combined"]]
import curious_snake
# todo: make learner setup function parameteric so you can pass it in here

#curious_snake.run_experiments_hold_out(feature_sets, os.path.join("output", "copd_combined_ODDS_RATIO_COTESTING"), num_runs=10, upto=800, hold_out_p=.25)
curious_snake.run_passive_mv_experiments(feature_sets, os.path.join("output", "copd_combined_CV_svm"))
