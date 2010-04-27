import os
import curious_snake

for d in ["r%s_denish" % (x+1) for x in range(14, 23)]:
    data_paths = [os.path.join("data", "sleep_apnea", "%s" % d,s) for s in ["sleep_titles", "sleep_abstracts", "sleep_keywords"]]#, "sleep_concepts"]]
    # todo: make learner setup function parameteric so you can pass it in here
    curious_snake.prospective(data_paths, os.path.join("output", "prospective_sleep_%s" % d), "predictions")