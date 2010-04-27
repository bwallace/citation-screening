import pdb

def all_concepts(file, threshold=800):
    concept_dict = {}
    all_concepts = []
    cur_doc = None
    for line in open(file, 'r').readlines():
        if line.startswith("Processing"):
            this_doc = line.split(" ")[1].split(".")[0]
            if this_doc != cur_doc:
                cur_doc = this_doc
                concept_dict[cur_doc] = []
        elif isint(line.strip().split(" ")[0]):
            concept_line = line.strip().split(" ")
            if eval(concept_line[0]) >= threshold:
                cur_concept = "-".join(concept_line[1:]).replace("\n", "")
                all_concepts.append(cur_concept)
                concept_dict[cur_doc].append(cur_concept)
    
    return (concept_dict, list(set(all_concepts)))
    
def isint(s):
    try:
        x = int(s)
    except:
        return False
    return True
    
def bag_o_concepts(concept_dict, all_concepts, at_least=3):
    filtered_concepts = [concept for concept in all_concepts\
                                     if concept_occurs_k_times(concept, concept_dict, at_least)]
    fout = open("concepts.txt", 'w')
    fout.write(str(filtered_concepts))
    fout.close()
    
    ids_to_feature_vecs = {}
    for doc_id, doc_concepts in concept_dict.items():
        cur_feature_vec = [0 for x in filtered_concepts]
        for i, concept in enumerate(filtered_concepts):
            if concept in doc_concepts:
                cur_feature_vec[i] = 1
        ids_to_feature_vecs[doc_id] = cur_feature_vec
    return ids_to_feature_vecs
            
        
    
def concept_occurs_k_times(concept, concept_dict, k):
    cur_count = 0
    for doc_concepts in concept_dict.values():
        if concept in doc_concepts:
            cur_count+=1
            if cur_count >= k:
                return True
    return False