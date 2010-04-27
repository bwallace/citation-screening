import pdb

p_index = 3 # where does the point start (i.e., after id/labels)

def join_files(paths, outpath):
    fvs = [open(path, 'r').readlines() for path in paths]
    all_fvs = [[l.replace("\n", "").split(" ") for l in fv] for fv in fvs]
    joined = join_fvs(all_fvs, get_maxes(all_fvs))
    pdb.set_trace()
    joined = [" ".join(ls) for ls in joined]
    out = open(outpath, 'w')
    out.write("\n".join(joined))
    out.close()
    
def get_maxes(fvs):
    dicts = []
    for fv in fvs:
        fv_max = 0
        for l in fv:
            dict = exec(fv[3:])
            cur_max = max(dict.keys())
            if  cur_max > fv_max:
                fv_max = cur_max 
        dicts.append(fv_max)
    return fv_max
        
        
def join_fvs(all_fvs, maxes):
    joined_fvs = all_fvs[0]
    for fv_file in all_fvs[1:]:
        for i in range(len(joined_vs)):
            cur_dict = exec("{"+fv_file[3:]+"}")
            joined_fv.append(fv[3:])
    return joined_fv 