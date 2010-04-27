

def labeled_feature_vecs(data, labeled_terms):
    ids_to_vecs = {}
    for x_id in data.keys():
        ids_to_vecs[x_id] = create_vec(data[x_id], labeled_terms)
    return ids_to_vecs
        
    
def create_vec(doc, labeled_terms):
    text = doc[0].lower() + " " + doc[1].lower()
    vec = []
    for i,term in enumerate(labeled_terms):
        if term in text:
            vec.append(1)
        else:
            vec.append(0)
    return vec
    

    
  
    
    
