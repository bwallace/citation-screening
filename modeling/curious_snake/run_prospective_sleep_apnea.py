######################################## 
#
# run hold out experiments on sleep apnea
#
########################################

import os
#feature_sets = [os.path.join("data", "sleep_apnea",s) for s in ["sleep_titles", "sleep_abstracts", "sleep_keywords", "sleep_concepts"]]
data_paths = [os.path.join("data", "sleep_apnea", s) for s in ["titles_post_r7", "abstracts_post_r7", "keywords_post_r7"]]
import curious_snake
import dataset
# todo: make learner setup function parameteric so you can pass it in here
datasets = [dataset.build_dataset_from_file(f) for f in data_paths]
curious_snake.prospective(None, os.path.join("output", "sleepies6"), "predictions_all", datasets=datasets, beta=1)
#curious_snake.retro_diversity(feature_sets)
