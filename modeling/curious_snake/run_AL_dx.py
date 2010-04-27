######################################## 
#
# run hold out experiments on ma_dx tests
#
########################################

import os
feature_sets = [os.path.join("data", "ma_dx",s) for s in ["ma_dx_titles", "ma_dx_abstracts", "ma_dx_keywords"]]
#feature_sets = [os.path.join("data", "sleep_apnea",s) for s in ["sleep_concepts"]]
import curious_snake
curious_snake.prospective_active_learn(feature_sets, os.path.join("output", "_AL", "dx_label_these101_120.csv"))
