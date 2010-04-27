######################################## 
#
# Active Learn
#
########################################

import os
feature_sets = [os.path.join("data", "sleep_apnea",s) for s in ["sleep_titles", "sleep_abstracts", "sleep_keywords", "sleep_concepts"]]
import curious_snake
# todo: make learner setup function parameteric so you can pass it in here
curious_snake.prospective_active_learn(feature_sets, os.path.join("output", "_AL", "sleep_label_these481_500.csv"))
