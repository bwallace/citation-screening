import os
feature_sets = [os.path.join("data", "sleep_apnea", "1000_labeled_only", s) for s in ["sleep_titles", "sleep_abstracts", "sleep_keywords"]]
import curious_snake
# todo: make learner setup function parameteric so you can pass it in here
curious_snake.run_experiments_hold_out(feature_sets, os.path.join("output", "SA_cofl"), num_runs=10, upto=1000, hold_out_p=.1)