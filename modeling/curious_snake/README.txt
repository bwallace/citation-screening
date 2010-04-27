******************************************************
Byron Wallace
Tufts Medical Center
---
Semi-Automated Citation Screening for Systematic Reviews:
Running the Experiments
******************************************************

Introduction
---
The experiments in the article were run using an open source active-learning framework we have developed called "curious snake". The library is written in Python (v2.5), atop a modified version of the C++ libSVM library. Here we have modified the library for the particulars of the citation screening scenario.


Directory Structure
---
The top-level directory is 'curious_snake'. This directory contains the script for running experiments (explained in the upcoming Running Experiments subsection). Underneath this directory are a number of packages used by the library and a directory called "data". This directory contains the datasets used in the article. 


Installation of the Code
---
Curious snake is cross platform, but it requires libSVM to be rebuilt on the client machine. The authors have succesfully done this on both mac os x and Windows machines. libSVM can be rebuilt by invoking "make" in the curious_snake top-level (this) directory. If the build completes with no problems, you should be able to import the "svm.py" module in the "learners/svm_learners/libsvm/python" directory (you can also check the svm_test.py module to see if everything is working).

The code for citation screening also also requires maplotlib to generate the yield/burden curves, which can be acquired at http://matplotlib.sourceforge.net/.


Running Experiments
---
The "run_experiments_finite_pool" method is the relevant routine for the citation screening experiments. It is only required to provide this method paths to the datasets (one per feature space) and the desired output path. Suppose we are interested in running experiments over the proton beam dataset. Further suppose we have started the Python interpreter in the top level directory (the same as this file). Then we can type the following:

> import os
> feature_sets = [os.path.join("data", "proton_beam", s) for s in ["keywords", "titles", "title_concepts"]]

The second line just builds a list of strings, each of which points to a data file pertaining to a particular feature-space. Note that here we omit "abstracts", but this file could also be included.

Now we can start the experimental run. At the Python console:

> import curious_snake
> curious_snake.run_experiments_finite_pool(feature_sets, os.path.join("output", "test"), upto=2000, num_runs=3)

Here we have specified to run our finite pool experiments 3 times (this is the num_runs parameter; by default it's 10) with up to 2000 labels over the datasets specified in our feature_sets vector. (For further explanations of the options/parameters, consult the inline documentation of the run_experiments_finite_pool method in curious_snake.py.) Further, we have told the program to write the output to the "output/test" directory. That's all we have to do; the program will run 3 experiments and subsequently generate the yield/burden curves as shown in the article.
