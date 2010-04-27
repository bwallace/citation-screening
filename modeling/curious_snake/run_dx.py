######################################## 
#
# run hold out experiments on ma_dx tests
#
########################################

import os
import dataset
#data_paths = [os.path.join("data", "ma_dx", "iter6_ndidi_1120" ,s) for s in ["ma_dx_titles60", "ma_dx_abstracts", "ma_dx_keywords"]]#, "ma_dx_concepts"]]
#test_data_paths =
data_paths= [os.path.join("data", "ma_dx", "iter2_ndidi_1040" ,s) for s in ["ma_dx_titles", "ma_dx_abstracts", "ma_dx_keywords"]]
datasets = [dataset.build_dataset_from_file(f) for f in data_paths]
#feature_sets = [os.path.join("data", "sleep_apnea",s) for s in ["sleep_concepts"]]
import curious_snake
# todo: make learner setup function parameteric so you can pass it in here
#curious_snake.retro_diversity(feature_sets)

#curious_snake.run_passive_mv_experiments(data_paths,
#                                                    os.path.join("output", "ma_dx_cv1000"), 
#                                                    num_runs=10, hold_out_p=.10)
                                                    
#curious_snake.run_cv_experiments_with_test_data(data_paths, test_data_paths,
#                                                    os.path.join("output", "ma_dx_cv1120"), 
#                                                    num_runs=10, hold_out_p=.10)
curious_snake.prospective(None, os.path.join("output", "dx_prospective_1040"), "dx_preds", datasets=datasets)

