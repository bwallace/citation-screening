'''
Byron C. Wallace 

Module for outlier detection
'''


def get_outliers():
    pass
    
def write_these_out(instances, path):
    max_dim = max([max(x.point.keys()) for x in instances])
    out = [get_str(inst, max_dim) for inst in instances]
    fout = open(path, 'w')
    fout.write("\n".join(out))
    fout.close()
    
def get_str(inst, max_dim):
    return str(inst.id) + ", " + ",".join(_map_x(inst.point, max_dim))
    
def _map_x(x, max_dim):
    x_prime = []
    # first, make sure x has the same dimensionality as the model
    for dim in range(max_dim):
        if dim not in x:
            x_prime.append("0.0")
        else:
            x_prime.append(str(x[dim]))
    return x_prime


def naive_detect(points, n, subspace=4, num_tries =10000):
    best_n = {}
    for step in num_tries:
        pass    
    
    