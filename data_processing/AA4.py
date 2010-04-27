import elementtree
from elementtree.ElementTree import ElementTree
import pdb
import random
import string
import pickle
import math
import random
import os


'''
Automatic Abstract Screening code.

This module deals with converting abstract data to feature vectors. Also includes SMOTEing (optionally) over the
minority (positive) class.
'''

def pickKRandomElementsFromList(l, k):
    newL = []
    alreadyPicked = []

    if k > len(l):
      pdb.set_trace()
      return None
    
    while len(newL) < k:
      i = random.randint(0, len(l)-1)
      if not i in alreadyPicked:
        newL.append(l[i])
        alreadyPicked.append(i)
    
    return newL


def tdidf(wordfreqs, freqvecs):
  '''
  Returns tf-idf feature vectors. For a simple explanation, see: http://instruct.uwo.ca/gplis/601/week3/tfidf.html

  wordfreqs -- Vector s.t. w[i] is the total number of times w[i] was seen over all documents.
  freqvecs -- A dictionary mapping document ids to frequency vectors. 
  
  returns a dictionary mapping the keys in freqvecs to their tf-idf feature-vector representation
  '''
  N = len(freqvecs.keys()) # Total number of documents
  num_terms = len(wordfreqs) # Number of terms
  print "Number of documents: %s, number of terms %s" % (N, num_terms)
  print "Building n_vec..."

  #
  # i is the document index; j the word/term index
  #

  print "Building n_vec..."
  n_vec = [0 for j in range(num_terms)]
  for i in range(N):
    cur_doc = freqvecs[freqvecs.keys()[i]]
    for j in range(num_terms):
        if cur_doc[j] > 0:
            n_vec[j]+=1
  
  for ind in range(len(n_vec)):
      if n_vec[ind] == 0:
	pdb.set_trace()
		  
  print "n_vec built."
  print "Now constructing TDF vector."

  tdfvecs = {}
  last_key = None
  for i in range(N):
    cur_key = freqvecs.keys()[i]
    last_key = cur_key
    
    if i%100 == 0:
        print "On document %s" % i

    cur_doc = freqvecs[cur_key]
    tdfvec = [0 for k in range(num_terms)] 
    for j in range(num_terms):
        tdfvec[j] = cur_doc[j] * math.log(N/n_vec[j], 2.0)
    # Normalize
    cos_norm = math.sqrt(sum([tdfvec[j]**2 for j in range(num_terms)]))
    
    if cos_norm == 0:
	# None of the terms were in thid document. Just return a vector of zeros.
	tdfvec = [0 for i in range(num_terms)]
    else:
        tdfvec = [tdfvec[j]/cos_norm for j in range(num_terms)]

    if cur_key in tdfvecs:
	print "key already exists???"
	pdb.set_trace()
    tdfvecs[cur_key]=tdfvec
  
  return tdfvecs


def cleanUpTxt(doc, stopListPath=None, keep=string.ascii_letters):
    ''' Cleans and returns the parametric abstract text. I.e., strips punctuation, etc. Also removes
    any words in the stop list (if provided).'''
    words = []
    excludeWords = []

    # Building stop words vector, if provided
    if stopListPath != None:
        f = open(stopListPath, 'r')
        while 1:
            line = f.readline()
            if not line:
                break
            # Every line is assumed to be a single word
            excludeWords.append(line.strip())
            
    for word in doc:
        clean_word = ''.join(c for c in word if c in keep)
        if clean_word and not clean_word in excludeWords:
            words.append(clean_word)

    return words

    
def xmlToDict(fPath, stopPath=None, splitTxt= False, get_pubmed = False):
    '''
    Converts study data from (ref man generated) XML to a dictionary matching study IDs (keys) to 
    title/abstract tuples (values). For example: dict[n] might map to a tuple [t_n, a_n] where t_n is the
    title of the nth paper and a_n is the abstract
    '''
    
    refIDToAbs = {}
    numNoPubmeds = 0
    numNoAbs = 0 # Keep track of how many studies have no abstracts.
    tree = ElementTree(file=fPath)
    
    for record in tree.findall('.//record'):
            pubmed_id = None
	    refmanid = eval(record.findall('.//rec-number')[0].text)
	    
            try:
                pubmed = record.findall('.//notes/style')[0].text
                pubmed = pubmed.split("-")
                for i in range(len(pubmed)):
                        if "UI" in pubmed[i]:
			    pubmed_str = pubmed[i+1].strip()
			    pubmed_id = eval("".join([x for x in pubmed_str if x in string.digits]))
                            #pubmed_id = eval(pubmed[i+1].replace("PT", "").replace("IN", ""))
                            #print pubmed
                            break
            except Exception, ex:
                print ex
                    
          
	    if pubmed_id is None:
		    #if not "Cochrane" in pubmed[2]:
		    #	pdb.set_trace()
		    numNoPubmeds+=1
		    print "%s has no pubmed id" % refmanid

            abstract = record.findall('.//abstract/style')
	    abText = ""
	    try:
		    if abstract and splitTxt:
			    abText = (abstract[0].text).split(" ")
			    abText = [string.lower(s) for s in abText]
			    abText = cleanUpTxt(abText, stopListPath=stopPath)
		    elif abstract:
			    abText = abstract[0].text
		    else:
			    numNoAbs += 1
	    except:
		    pdb.set_trace()


	    title = ""
	    if splitTxt:
	            title = cleanUpTxt(string.lower(record.findall('.//titles/title/style')[0].text).split(" "), stopListPath=stopPath)
	    else:
		    try:
			title = record.findall('.//titles/title/style')[0].text
		    except:
			pdb.set_trace()
		    

	    # Also grab keywords
	    keywords = [keyword.text.strip().lower() for keyword in record.findall(".//keywords/keyword/style")]
	    if pubmed_id is not None or True:
                     refIDToAbs[refmanid] = [title, abText, keywords, pubmed_id]
	    #pdb.set_trace()
    print "Finished. Returning %s title/abstract/keyword sets, %s of which have no abstracts, and %s of which have no pubmed ids." % (len(refIDToAbs.keys()), numNoAbs, numNoPubmeds)
    return refIDToAbs

def dict_to_metamap(d, out_file):
    '''
    e.g.,
    0000001|item 1 text to be processed free text with ID
    0000002|item 2 text to be processed.
    '''
    out_str = []
    for id, texts in d.items():
        # assuming texts contains: [title, abstract, keywords]
        out_str.append("%s|%s" % (id, " ".join([texts[0], texts[1]])))
    outf = open(out_file, 'w')
    outf.write("\n".join(out_str))
    outf.close()

def write_out_text(d, index, out_dir):
    for d_id, text_vals in d.items():
        fout = open(os.path.join(out_dir, str(d_id)), 'w')
	if isinstance(index, list):
	    fout.write(" ".join([d[d_id][i] for i in index]))
	elif isinstance(d[d_id][index], list):
	    fout.write(" ".join(d[d_id][index]))
	else:
	    fout.write(d[d_id][index])
	fout.close()

def buildWordListFromDoc(doc):
    '''
    Builds and returns a vector containing all the words in doc (stripped of punctuation).
    '''
    ls = [word for word in doc]
    #unpacked_ls = []
    #for entry in ls:
    #    if type(entry) == type([]):
    #        unpacked_ls.extend(entry)
    #    else:
    #        unpacked_ls.append(entry)
    #return unpacked_ls
    return ls
    
    
def buildWordList(docs, k=3):
    '''
    Returns a dictionary with keys set to all words found in the docs list.
    '''
    #pdb.set_trace()
    print "working on a doc?"
    wList = []
    i = 0
    total_docs = len(docs)
    for doc in docs:
	if i%500 == 0:
            print "on doc %s out of %s" % (i, total_docs)
        i+=1
        docList = buildWordListFromDoc(doc)
        if docList == None:
            pdb.set_trace()
        wList.extend(docList)
	
    #wList = list(set(wList))
    #pdb.set_trace()
    # Only use words that appear at least k times
    sigWords = []
    count = 0
    print len(set(wList))
    for word in set(wList):
      if count % 500 == 0:
	print count
      count+=1
      if wList.count(word) >= k:
        sigWords.append(word)
    
    #wList = [word for word in wList if wList.count(word) >= k]
    wList = [word for word in set(wList) if word in sigWords]
    
    return list(set(wList))


def buildWordCountVectorForDoc(words, doc):
    '''
    Returns a vector V where V_i corresponds to the number of times words_i is contained in doc
    '''
    countVec = [0 for i in range(len(words))]

    for i in range(len(words)):
      for word in doc:
        if word == words[i]:
          countVec[i]+=1
    
    return countVec


def buildFeatureVectors(studies, keywords):
	abstractFeatureVect = {} 
	titleFeatureVect = {}
	
	# We need to keep track of the total word counts for our keywords
	totalWordCountsInAbstracts = [0 for word in keywords] 
	totalWordCountsInTitles = [0 for word in keywords]

	for study_id in studies.keys():
	    # abstracts
	    curAbsWordCountVect = buildWordCountVectorForDoc(keywords, studies[study_id][1])
	    abstractFeatureVect[study_id] = curAbsWordCountVect
	    totalWordCountsInAbstracts = addLists(totalWordCountsInAbstracts, curAbsWordCountVect)
	    
	    # titles
	    curTitleWordCountVect = buildWordCountVectorForDoc(keywords, studies[study_id][0])
	    titleFeatureVect[study_id] = curTitleWordCountVect
	    totalWordCountsInTitles = addLists(totalWordCountsInTitles, curTitleWordCountVect)
	    
	
	# Now construct td-idf vectors
	abstractTDIDF = tdidf(totalWordCountsInAbstracts, abstractFeatureVect)
	titlesTDIDF = tdidf(totalWordCountsInTitles , titleFeatureVect)
	
	return [abstractTDIDF, titlesTDIDF]
	
	
def buildBagOfWordsFeatureVector(studies, words, vectorIndex=1):
    freqVecs = {}
    for id in studies.keys():
        freqVecs[id] = buildWordCountVectorForDoc(words, studies[id][vectorIndex])
	
    wordFreqs = [0 for i in range(len(words))]
    for doc_index in studies.keys():
        for word_index in range(len(freqVecs[doc_index])):
             wordFreqs[word_index]+=freqVecs[doc_index][word_index]

    return tdidf(wordFreqs, freqVecs)

	
	
def pickKRandomElementsFromLists(l1, l2, k, labels):
  '''
  Warning -- This deletes all the items picked from the list l!
  '''
  print "Warning-- Deleting selected items from list!"
  newL = []
  newLabels = []

  if k > len(l):
      return None
  
  while len(newL) < k:
      i = random.randint(0, len(l)-1)
      
      newL.append(l[i])
      newLabels.append(labels[i])
      del l[i]
      del labels[i]
  
  return [newL, newLabels]


def cautiousClassifyWithMultipleSVMs(svms, test_sets, true_lbls, thresholds):
	'''
	If any one of the classifiers says 'yes'; we vote yes.
	
	We will store results as follows. Let the rows correspond to our predicted class and columns correspond to the true class,
	so that m[0,1] gives the number of instances for which we predicted 0 but whose actual label is 1.
	
		-1 	1
	-1       x  	y
	 1       z  	n
	
	x = true negatives
	z = false positive
	y = false negative <- must minimize this, in particular!
	n = true positive
	'''
	confusion_matrix = [[0,0],[0,0]]
	print "using %s classifiers..." % len(svms)
	
	# Now classify
	for i in range(len(test_sets[0])):
	     # predic_probability returns a tuple. The first entry is the predicted label; the second is a dictionary comprised of the
	     # respective probabilities that this instance belongs to the two classes.
	     # For each example, we apply each model. Each model is assumed to have a corresponding test_set
	     all_probs = [svms[model_index].predict_probability(test_sets[model_index][i]) for model_index in range(len(svms))]
             pdb.set_trace()
	     prediction_ind = 0 # 0 corresponds to predicting -1
	     
	     # If any of the classifiers say yes, we vote yes.
	     for model_index in range(len(all_probs)):
		 probs = all_probs[model_index]
	         # If there is at least a thresh% chance (on our model) that this instance is positive (relevant)
	         # predict 1.
	         if probs[1][1] >= thresholds[model_index]:
	             prediction_ind = 1 # corresponds to predictiong 1
	     
	     # the column corresponds to the actual value
	     col = 0
	     if true_lbls[i]==1:
	         col = 1
	     
	     confusion_matrix[prediction_ind][col] += 1
	     
	print "Done classifying. Here's your confusion matrix:\n"
	prettyPrintConfusionMatrix(confusion_matrix)
	return confusion_matrix	


def splitIntoTrainAndTestSets(k,all_insts, all_lbls, SMOTE=False):
	train_set_indices = PyUtils.pickKRandomThings(range(len(all_lbls)), k)
	all_train_sets, all_test_sets = [], []
	train_lbls, test_lbls = [], []
	lbls_built = False
	
	for inst_set in all_insts:
	    train_set, test_set = [], []
	    for i in range(len(inst_set)):
	        if i in train_set_indices:
	            if not lbls_built:
		        train_lbls.append(all_lbls[i])
		    train_set.append(inst_set[i])
		else:
		    if not lbls_built:
		        test_lbls.append(all_lbls[i])
		    test_set.append(inst_set[i])
	    lbls_built = True
	    all_train_sets.append(train_set)
	    all_test_sets.append(test_set)
	return [all_train_sets, all_test_sets, train_lbls, test_lbls]

	
	
	
def buildModel(insts, train_lbls):
	param = svm_parameter()
        param.probability = 1
	#param.C = 100
        prob = svm_problem(train_lbls, insts)
        m = svm_model(prob, param)    
	return m
	    
	    
def prettyPrintConfusionMatrix(confusion_matrix):
	print "         \t\tTRUE"
	print "PREDICTED\t-1\t1"
	print "       -1\t" + str(confusion_matrix[0][0]) + "\t" + str(confusion_matrix[0][1])
	print "        1\t" + str(confusion_matrix[1][0]) + "\t" + str(confusion_matrix[1][1])


def addLists(l1, l2):
	if len(l1) != len(l2):
		print "List sizes unequal!"
		return None
	return [l1[i]+l2[i] for i in range(len(l1))]
		
	
def makeLblsAndInsts(all_insts, included):
	lbls = []
	insts = []
	for key in all_insts.keys():
		if key in included.keys():
			lbls.append(1)
		else:
			lbls.append(-1)
		insts.append(all_insts[key])
	return [insts,lbls]
    


def buildTrainTestSets(freqVecs, posList, negList, numPosExamples, numNegExamples, SMOTEing=True, N=500):
    '''
    The freqVecs is a dictionary of all document frequency vectors. The pos and neg lists are the titles for the positive and negative
    abstracts. 
    
    Returns a four-tuple [t, l, t', l'] where [t,l] comprise the the training instances and labels, respectively and [t',l']
    are the testing instances and labels. Not that |t| = numPosExamples + numNegExamples and |t'| = |freqVecs| - (numPosExamples+numNegExamples)
    The freqVecs s
    '''
    print "Building training set..."
    trainingFreqVecsList = []
    trainingLabels = []
    posSoFar = negsSoFar = 0
    posFreqVecs = [] # Keep this list for SMOTEing purposes
    while len(trainingLabels) < (numPosExamples + numNegExamples):
        title = pickKRandomElementsFromList(freqVecs.keys(), 1)[0]
        if title in posList and posSoFar < numPosExamples:
            # Add to the training set, delete from freqVecs
            trainingFreqVecsList.append(freqVecs[title])
            posFreqVecs.append(freqVecs[title])
            trainingLabels.append(1)
            posSoFar += 1
            del freqVecs[title]
        elif title in negList and negsSoFar < numNegExamples:
            trainingFreqVecsList.append(freqVecs[title])
            trainingLabels.append(0)
            negsSoFar += 1
            del freqVecs[title]
  
    if SMOTEing:
        # If we're SMOTEing, add synthetic examples
        print "holy smotes. SMOTEing with %s percent." % N
        synthetic = smote.SMOTE(posFreqVecs, N)
        print "SMOTEing: Adding %s synthetic positive examples." % len(synthetic)
        for inst in synthetic:
            trainingFreqVecsList.append(inst)
            trainingLabels.append(1)

    # Here we build the test set. Note that we deleted all the instances used in the 
    # training set
    print "Done. Building test set..."
    freqVecsList = []
    labels = [] 
    for title in freqVecs.keys():
        freqVecsList.append(freqVecs[title])
        if title in posDict.keys():
            labels.append(1)
        else:
            labels.append(0)

    print "Test set constructed."
    return [trainingFreqVecsList, trainingLabels, freqVecsList, labels]


''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'      WEKA (ARFF) data mining format.                                    '
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def buildWekaLineStr(label, wordFreq):
    '''
    Create a WEKA style (ARFF) line for the document associated with the provided wordFreq parameter.
    '''
    line = ["{"]
    for i in range(len(wordFreq)):
        # Sparse formatting: Give the attribute 'index' first, then the value if it's non zero
        if wordFreq[i] > 0:
            line.append(str(i) + " " + str(wordFreq[i]))
    return ", ".join(line) + ", " + str(len(wordFreq)) + " " + str(label) + "}"


def generateWekaFile(labels, frequencyVectors, words, outPath):
    '''
    Builds and writes out a WEKA formatted file with the word frequencies as attributes for each instance.
    '''
    wekaStr = ["@RELATION abstracts"]
    for i in range(len(words)):
        # e.g.,: @ATTRIBUTE sepallength NUMERIC
        wekaStr.append("@ATTRIBUTE " + words[i] + " INTEGER")
    wekaStr.append("@ATTRIBUTE class {0,1}")
    wekaStr.append("\n@DATA")
    for instance in range(len(frequencyVectors)):
        wekaStr.append(buildWekaLineStr(labels[instance], frequencyVectors[instance]))
    fOut = open(outPath, "w")
    fOut.write("\n".join(wekaStr))
    


def dictToFreqVecs(fVecDict, posTitles):
  X, lbls, patternIDs = [], [], []
  for title in fVecDict.keys():
    X.append(fVecDict[title])
    patternIDs.append(title)
    # Now append the label
    if title in posTitles:
      lbls.append("1")
    else:
      lbls.append("0")
  return (X, lbls, patternIDs)
        
    
def parse_data():
    allPath = "data/all_XML.xml"
    positives = "data/positive_XML.xml"

    # allDict contains

    '''
    allDict = xmlToDict(allPath, "data/stopWords.txt")
    posDict = xmlToDict(positives, "data/stopWords.txt")
    # Create our negative dictionary (ALL-POSITIVES)
    negDict = allDict.copy()
    for title in posDict.keys():
        del negDict[title]
    
    '''

    '''
    print "pickling..."
    pickle.dump(allDict, open("data/alldict.pickle", 'w'))
    pickle.dump(posDict, open("data/posdic.pickle", 'w'))
    pickle.dump(negDict, open("data/negDict.pickle", 'w'))
    print "Done."
    '''

    print "unpickling allDict..."
    allDict = pickle.load(open("data/alldict.pickle", 'r'))
    print "unpickling posDict..."
    posDict = pickle.load(open("data/posdic.pickle", 'r'))
    print "unpickling negDict..."
    negDict = pickle.load(open("data/negdict.pickle", 'r'))
    print "done unpickling."
    
    print "Total number of abstracts: %s\n 'Positive' examples: %s\n 'Negative' examples: %s\n" % (len(allDict.keys()), len(posDict.keys()), len(negDict.keys()))

    # Now we build a word list over all abstracts                                
    #
    # The commented out bit was redundantly appending the 'positive' examples to the 'master' list. The positive examples
    # are already included in the allDict.
    allAbstractsList = [allDict[title] for title in allDict.keys()] #+ [posDict[title] for title in posDict.keys()] 
    #allWords = buildWordList(allAbstractsList)
    allWords = pickle.load(open("words_list.txt", 'r'))

    print "Words list built. Found %s words." % len(allWords)
    #print "Pickling words list..."
    #pickle.dump(allWords, open("words_list.txt", 'w'))
    print "Building word count vectors..."

    '''
    print "Building frequency vectors for all documents..."
    freqVecs = {}
    for title in allDict.keys():
        freqVecs[title] = buildWordCountVectorForDoc(allWords, allDict[title])
    pdb.set_trace()
    pickle.dump(freqVecs, open("freqvecs.pick", 'w'))
    '''
    #
    freqVecs = pickle.load(open("freqvecs.pick", 'r'))
    #

    #
    # td-idf!
    #

 
    wordFreqs = [0.0 for i in range(len(allWords))]
    for doc_index in freqVecs.keys():
      for word_index in range(len(freqVecs[doc_index])):
        wordFreqs[word_index]+=freqVecs[doc_index][word_index]

    '''
    print "Building td-idf feature vectors..."
    tdf_freq_dict =  tdidf(wordFreqs, freqVecs)
    print "Success."
    pickle.dump(tdf_freq_dict, open("tdf_freqvecs.pickle", 'w'))
    print "Pickled and dumped!" 
    '''
    print "Unpickling frequency vectors"
    tdf_freq_dict = pickle.load(open("tdf_freqvecs.pickle", 'r'))
    print "done."
    

    #
    # Splitting data into training/test sets. Want to oversample positive examples for training.
    # Note that in general ~10% of the abstracts are 'positive' (i.e., relevant)
    #
    '''
    numPos = 15
    numNeg = 60
    [trainingFreqVecsList, trainingLabels, freqVecsList, labels] = buildTrainTestSets(freqVecs, posDict.keys(), negDict.keys(), numPos, numNeg)
    

    print "Building WEKA (arff) files..."
    generateWekaFile(trainingLabels, trainingFreqVecsList, allWords, "train.arff") 
    generateWekaFile(labels, freqVecsList, allWords, "test.arff") 
    print "Finished."
    '''
    # This is a three-tuple: See in-line comments in dictToFreqVecs
    return dictToFreqVecs(tdf_freq_dict, posDict.keys())

def run_exps(X, lbls, patternIDs, K):
	results = {}
	for k in K:
		train_s = X[:k]
		train_lbls = lbls[:k]
		# We Smote the positive examples.
		pos_ex = [train_s[i] for i in range(len(train_s)) if lbls[i] == '1'] 
		print "Smoting"
		synthetic = smote.SMOTE(pos_ex, 300)
		for ex in synthetic:
			train_s.append(ex)
			train_lbls.append('1')
		
		train_d = datafunc.VectorDataSet(train_s,L=train_lbls)
		test_d = datafunc.VectorDataSet(X[k:],L=lbls[k:],patternID=patternIDs[k:])
		print "Training SVM..."
		s = svm.SVM()
		s.train(train_d)
		print "done. Testing..."
		r = s.test(test_d)
		results[k] = r.getConfusionMatrix()
	return results
	
if __name__ == "__main__":
    X, lbls, patternIDs = parse_data()
    k = 200
    t1 = X[:k]
    pos = [t1[i] for i in range(len(t1)) if lbls[i] == '1']
    train_d = datafunc.VectorDataSet(X[:k],L=lbls[:k],patternID=patternIDs[:k])
    test_d = datafunc.VectorDataSet(X[k:],L=lbls[k:],patternID=patternIDs[k:])
    s = svm.SVM()
    print "Training SVM..."
    s.train(train_d)
    
def find_threshes(expected_num_positives, test_sets, classifiers, max_iters=10000):
    return [find_threshold(classifiers[classifier_index], test_sets[classifier_index], expected_num_positives, max_iters=max_iters) for classifier_index in range(len(classifiers))]
	    
	    
def find_threshold(classifier, test_set, expected_num_positives, max_iters=10000, acceptable_diff=5):
    thresh = 0.5
    last = thresh
    found_positives, iter = 0, 0
    #pdb.set_trace()
    
    # allow 'overshooting' i.e., acceptable_diff more than expected
    while found_positives-expected_num_positives <= acceptable_diff and iter < max_iters:
	if iter % 50 == 0:
	    print "on iteration %s, current threshold %s" % (iter, thresh)
	    print "cur dif is %s\n" % str(found_positives-expected_num_positives)
        found_positives = 0
	for example in test_set:
            if classifier.predict_probability(example)[1][1] >= thresh:
                found_positives+=1
	
	# Adjust threshold
	#pdb.set_trace()
	last = thresh
        if found_positives <  expected_num_positives:
            # then lower the threshold
	    thresh = thresh - .5 * thresh
	else:
	    # otherwise raise the threshold
	    thresh = thresh + .1 * thresh
	iter+=1

    if iter >= max_iters:
        thresh = min(thresh, last)
	
    print "returning threshold %s" % thresh
    return thresh
    
	
def write_out_instances_for_lib_svm(fname, instances, lbls):
    out_str = []
    cur_index = 0
    for inst in instances:
	out_str.append(str(lbls[cur_index]) + " " + " ".join([str(att_index+1) + ":" + str(inst[att_index]) for att_index in range(len(inst))]))
	cur_index+=1
    f = open(fname, 'w')
    f.write("\n".join(out_str))


def rock_out(k, threshes=None, num_runs=1):
    terms = open('proton_terms').readlines()[0].split(",")
    all = xmlToDict("all.xml", "stopWords.txt")
    pos = xmlToDict("1s.xml", "stopWords.txt")


    '''
    #
    # key words
    #
    allKeyWordsList = [all[title][2] for title in all.keys()]
    allKeyWords = buildWordList(allKeyWordsList)
    key_insts = buildBagOfWordsFeatureVector(all, allKeyWords, vectorIndex=2) 
    key_insts, key_lbls = makeLblsAndInsts(key_insts, pos)
    
    ## read in all words
    #open("allwords_new2.txt", "w").write(str(allWords))    
    '''
    allWords = eval(open("allwords_new2.txt").readlines()[0])
    
    
    #
    # bag of words
    #
   # removing this for now because it is PAINFULLY slow
   # 11/13/08 -- didn't improve classification, either
    #bag_insts = buildBagOfWordsFeatureVector(all, allWords, vectorIndex=1)
    #bag_insts, bag_lbls = makeLblsAndInsts(bag_insts, pos)
    # maybe pickle here? 
    
    '''
    #
    # abstracts and titles
    #
    feat_vecs = buildFeatureVectors(all, terms)
    # feat_vecs[0] abstracts; feat_vecs[1] titles
    abstract_insts, abstract_lbls = makeLblsAndInsts(feat_vecs[0], pos)
    title_insts, title_lbls = makeLblsAndInsts(feat_vecs[1], pos)
    print "done creating different instances !"
    
    # Now split train and test sets
    # Note that the lablels can remain the same
    # removing bag_insts for now
    all_instances = [abstract_insts, title_insts, key_insts]
    
    
    #print "pickling!"
   # f = open('all_instances', 'w')
    #pickle.dump(all_instances, f)
    #f.close()
    
    f = open('abstract_labels', 'w')
    pickle.dump(abstract_lbls, f)
    f.close()
    print "pickled!" 
    '''
    
    
    # or unpickle
    print "unpickling!"
    all_instances = pickle.load(open('all_instances', 'r'))
    abstract_lbls = pickle.load(open('abstract_labels', 'r'))
    print "done!"
    
    #all_instances.append(bag_insts)
    
    overall_confusion_matrix = [[0,0],[0,0]]
    for run in range(num_runs):
	print "\n\n on run: %s" % (run+1)
        #  abstract_insts, title_insts, key_insts
        #all_train_sets, all_test_sets, training_lbls, test_lbls = splitIntoTrainAndTestSets(k, all_instances, abstract_lbls)
	
	set_names = ["abstracts", "titles", "keywords"]
	for i in range(len(set_names)):
	    #pdb.set_trace()
	    #write_out_instances_for_lib_svm(set_names[i] + "_TRAIN_%s"%run, all_train_sets[i], training_lbls)
	    #write_out_instances_for_lib_svm(set_names[i] + "_TEST_%s"%run, all_test_sets[i], test_lbls)
	    write_out_instances_for_lib_svm(set_names[i] + "_ALL", all_instances[i], abstract_lbls)
      
	'''
        # here let's SMOTE before building models
        pos_example_indices = [i for i in range(len(training_lbls)) if training_lbls[i] == 1]
        print "found %s positive examples in training set... SMOTEing." % len(pos_example_indices)
        smoted_train_sets, smoted_training_lbls = all_train_sets, training_lbls
        for i in range(len(all_train_sets)):
	    train_set = all_train_sets[i]
	    synthetics = smote.SMOTE([train_set[x] for x in pos_example_indices], 500)
            smoted_train_sets[i].extend(synthetics)
	    # only need to do this once.
	    if i==0:
	        smoted_training_lbls.extend([1 for x in range(len(synthetics))])

        models = [buildModel(smoted_train_set, smoted_training_lbls) for smoted_train_set in smoted_train_sets]
	print "ws?"
        pdb.set_trace()
    
        #print "\n\n--- Classifying with ABSTRACTS ---\n"
        print "building model with abstracts..."
        #abs_model = buildModelWithKExamples(k, abstract_insts, abstract_lbls)
        #cautiousClassifyTrainOnK(k, abstract_insts, abstract_lbls, thresh=pos_thresh)
        #print "\n done. building model with titles..."
        #titles_model = buildModelWithKExamples(k, title_insts, title_lbls)
        #cautiousClassifyTrainOnK(k, title_insts, title_lbls, thresh=pos_thresh)
        print "finding thresholds..."
        #threshes = find_threshes(int(round(.2*len(all_train_sets[0]))), all_test_sets, models)
        #print "THRESHOLDS!"
        if not threshes:
            threshes = [.5 for x in range(len(models))]

        print threshes
        #pdb.set_trace()
        #.2, .07, .5 # ORIGINAL 
        #[0.13750000000000001, 0.068750000000000006, .4, 0.27500000000000002]
        #theshes = [0.13750000000000001, 0.068750000000000006, 0.40000000000000002, 0.275000] works pretty well!
        #threshes = [.15, .065, .4]
    
        res = cautiousClassifyWithMultipleSVMs(models, all_test_sets, test_lbls, threshes)
	for row in range(2):
	    for col in range(2):
                overall_confusion_matrix[row][col] += res[row][col]
    
    # average
    print "\n\nfinished with all %s runs, here is the averaged confusion matrix:" % num_runs
    for row in range(2):
        for col in range(2):
            overall_confusion_matrix[row][col] = overall_confusion_matrix[row][col]/float(num_runs)
	    
    prettyPrintConfusionMatrix(overall_confusion_matrix)
    '''
