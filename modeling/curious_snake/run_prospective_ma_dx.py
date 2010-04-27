import os

data_paths = [os.path.join("data", "ma_dx", "all", s) for s in ["ma_dx_titles", "ma_dx_abstracts", "ma_dx_keywords"]]
import curious_snake
import dataset
# todo: make learner setup function parameteric so you can pass it in here
datasets = [dataset.build_dataset_from_file(f) for f in data_paths]
curious_snake.prospective(None, os.path.join("output", "ma_dx"), "predictions_all", datasets=datasets)
#curious_snake.retro_diversity(feature_sets)
