######################################## 
#
# run hold out experiments on sleep apnea
#
########################################

import os
import dataset
#data_paths = [os.path.join("data", "sleep_apnea", "r0_denish",s) for s in ["sleep_titles", "sleep_abstracts", "sleep_keywords"]]#, "sleep_concepts"]]
#feature_sets = [os.path.join("data", "sleep_apnea",s) for s in ["sleep_concepts"]]
test_data_paths =  [os.path.join("data", "sleep_apnea", "r0",s) for s in ["sleep_titles", "sleep_abstracts", "sleep_keywords"]]#, "sleep_concepts"]]
test_datasets = [dataset.build_dataset_from_file(f, ignore_unlabeled_instances=True) for f in test_data_paths]
#feature_sets = [os.path.join("data", "sleep_apnea",s) for s in ["sleep_concepts"]]
import curious_snake
# todo: make learner setup function parameteric so you can pass it in here
#curious_snake.retro_diversity(feature_sets)

data_paths = [os.path.join("data", "sleep_apnea", "%s" % "r0_denish",s) for s in ["sleep_titles", "sleep_abstracts", "sleep_keywords"]]#, "sleep_concepts"]]
curious_snake.run_cv_experiments_with_test_data(data_paths, test_data_paths, 
                                                    os.path.join("output", "retro_cv_no_undersample"), 
                                                    test_datasets=test_datasets,
                                                    num_runs=10, hold_out_p=.10)
'''
for d in ["r0"] + ["r%s_denish" % (x+1) for x in range(23)]:
    print d
    data_paths = [os.path.join("data", "sleep_apnea", "%s" % d,s) for s in ["sleep_titles", "sleep_abstracts", "sleep_keywords"]]#, "sleep_concepts"]]
    curious_snake.run_cv_experiments_with_test_data(data_paths, test_data_paths, 
                                                        os.path.join("output", "retro_cv_no_undersample_%s" % d), 
                                                        test_datasets=test_datasets,
                                                        num_runs=10, hold_out_p=.10)
                                                    
'''
