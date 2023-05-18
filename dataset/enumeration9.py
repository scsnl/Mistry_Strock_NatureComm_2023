import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from nn_modeling.dataset.enumeration import *

seed = 0
np.random.seed(seed)

# -------------------------------
# General parameters
# -------------------------------

n_max = 9 # maximal number of dots 
ns = np.arange(1,n_max+1) # number of dots considered
img_dim = (224,224) # dimension of the pictures
img_a = img_dim[0]*img_dim[1] # total area of a picture
n_sample = 12 # number of sample per numerosity
n_parameters = 50 # number of parameters per conditio

# -------------------------------
# Color and shape parameters
# -------------------------------

color_d = 100 # min color distance between background and dot colors
gen_color = lambda n: random_color(n, color_d) # all dots in a picture have the same color
gen_shape = circle_shape # dots are circular

# -------------------------------
# Area parameters
# -------------------------------

r_min = 2
r_max = 50
a_min_one = np.pi*r_min**2 # min area of one dot
a_max_one = np.pi*r_max**2 # max area of one dot when randomly sampled
min_a_mean = n_max*a_min_one #100 # min mean total area for the dots in a condition
max_a_mean = a_max_one #5000 # max mean total area for the dots in a condition
a_max = n_max*2*max_a_mean/(1+n_max) # max total area (used for plots)
a_mean_all = np.random.uniform(min_a_mean, max_a_mean, n_parameters) # mean area considered

# -------------------------------
# Convex hull parameters
# -------------------------------

ch_d = 2 # min distance between dots
k = 0.5 # repulsion speed caused by dots overlapping
g = 0.5 # attraction/repulsion speed to the barycenter
min_move = 0.5 # minimum movement
stop = 100 # max number of step in a trial to obtain a correct configuration
n_try = 10 # number of trials to obtain a correct configuration
tolerance = 0.01 # relative tolerance criterion for the target convex hull
min_coverage_mean = 0.3 # min mean coverage of the convex hull on the image
max_coverage_mean = 0.5 # max mean coverage of the convex hull on the image
ch_max = n_max*2*max_coverage_mean*img_a/(1+n_max) # max convex hull area (used for plots)
coverage_mean_all = np.random.uniform(min_coverage_mean, max_coverage_mean, n_parameters) # mean convex hull area considered

# -------------------------------
# Saving paths
# -------------------------------

data_path = f'{os.environ.get("DATA_PATH")}/enumeration_{n_max}/stimuli'
figure_path = f'{os.environ.get("FIG_PATH")}/enumeration_{n_max}/stimuli-properties'
os.makedirs(f'{data_path}', exist_ok=True)
os.makedirs(f'{figure_path}', exist_ok=True)

# -------------------------------
# Conditions
# -------------------------------

# target areas as a function of the number of dots
target_area = {
    "fixed_total_area": lambda n,a_mean: a_mean,
    "linear_total_area": lambda n,a_mean: n*2*a_mean/(1+n_max)
}

# target convex hull areas as a function of the number of dots
target_convexhull = {
    "fixed_convexhull": lambda n, ch_mean: ch_mean,
    "linear_convexhull": lambda n, ch_mean: n*2*ch_mean/(1+n_max)
}

# size of the dots in a picture given the target area
size_method = {
    "fixed_size": lambda shape,a: area_controlled_fixed_size(shape, a),
    "random_size": lambda shape,a:  area_controlled_random_size(shape, a, a_min_one, a_max_one)
}

# -------------------------------
# Metrics
# -------------------------------

# Metric extracted from data
metric = {
    "success": lambda color, shape, size, location, success, img: success,
    "img_area": lambda color, shape, size, location, success, img: area_img(img, color[0]),
    "img_charea": lambda color, shape, size, location, success, img: convex_hull_img(img, color[0]),
    "exp_area": lambda color, shape, size, location, success, img: area_shape(shape, size),
    "exp_charea": lambda color, shape, size, location, success, img: convex_hull_shape(shape, size, location),
    "mean_radius": lambda color, shape, size, location, success, img: np.mean(size),
    "exp_perimeter": lambda color, shape, size, location, success, img: perimeter_shape(shape, size)
}
metric_id = list(metric.keys())

# Metric computed from other metrics
more_metric = {
    "img_area_%": lambda d: d["img_area"]/img_a,
    "img_charea_%": lambda d: d["img_charea"]/img_a,
    "img_density": lambda d: d["img_area"]/d["img_charea"],
    "exp_density": lambda d: d["exp_area"]/d["exp_charea"]
}
more_metric_id = list(more_metric.keys())

metric_name = {
    "success": "correct",
    "img_area": "picture dot area",
    "img_charea": "picture ch area",
    "exp_area": "expected dot area",
    "exp_charea": "expected ch area",
    "mean_radius": "mean radius of dots",
    "exp_perimeter": "expected dot perimeter",
    "img_density": "picture density",
    "exp_density": "expected density",
    "img_area_%": "dot coverage",
    "img_charea_%": "ch coverage"
}

# Maximum of the metric (used for plots)
metric_max = {
    "success": 1,
    "img_area": a_max,
    "img_charea": ch_max,
    "exp_area": a_max,
    "exp_charea": ch_max,
    "mean_radius": area_to_size('c', a_max),
    "exp_perimeter": 2*np.sqrt(n_max*a_max),
    "img_density": 1,
    "exp_density": 1,
    "img_area_%": 1,
    "img_charea_%": 1
}

metric_target = {
    "img_area": lambda ns, a_mean, ch_mean, condition_area, condition_location: [target_area[condition_area](n, a_mean) for n in ns],
    "img_charea": lambda ns, a_mean, ch_mean, condition_area, condition_location: [target_convexhull[condition_location](n, ch_mean) for n in ns],
    "exp_area": lambda ns, a_mean, ch_mean, condition_area, condition_location: [target_area[condition_area](n, a_mean) for n in ns],
    "exp_charea": lambda ns, a_mean, ch_mean, condition_area, condition_location: [target_convexhull[condition_location](n, ch_mean) for n in ns]
}
for m in metric_id+more_metric_id:
    if not m in metric_target.keys():
        metric_target[m] = lambda ns, a_mean, ch_mean, condition_area, condition_location: None


metric_f = {}
for m in metric_id:
    metric_f[m] = ".0f"
metric_f["success"] = ".0%"
for m in more_metric_id:
    metric_f[m] = ".0%"
metric_cor = ["img_area","img_charea", "mean_radius", "img_density"]

data = {
    f'{k1}_{k2}_{k3}': {
        m: []
        for m in metric_id+more_metric_id
    }
    for k1 in size_method.keys()
    for k2 in target_area.keys()
    for k3 in target_convexhull.keys()
}


# -------------------------------
# Plotting statistics
# -------------------------------

def plot_statistics(ns, data, metric_name, metric_target, metric_max):
    n_subplot = len(data)
    f = plt.figure(figsize = (4, 2*n_subplot))
    for i, (m,d) in enumerate(data.items()):
        mean_d = np.mean(d, axis = 1)
        std_d = np.std(d, axis = 1)
        ax = f.add_subplot(n_subplot,1,i+1)
        ax.ticklabel_format(scilimits=(-2,2))
        ax.set_ylabel(metric_name[m])
        ax.set_ylim(0,1.1*metric_max[m])
        ax.fill_between(ns, mean_d-std_d, mean_d+std_d, color = "0.7")
        ax.plot(ns, mean_d, color = "0.0")
        if not metric_target[m] is None:
            ax.plot(ns,metric_target[m], color = "red", linestyle = "--")
    ax.set_xlabel(f'number of dots')
    return f

# -------------------------------
# Generation loop
# -------------------------------

for i in tqdm(range(n_parameters)):
    a_mean = a_mean_all[i]
    ch_mean = coverage_mean_all[i]*img_a
    for condition_size in size_method.keys():
        for condition_area in target_area.keys():
            for condition_location in target_convexhull.keys():
                condition = f'{condition_size}_{condition_area}_{condition_location}'
                params = f'mean-area_{a_mean:.0f}_mean-charea_{ch_mean:.0f}'
                gen_size = lambda shape: size_method[condition_size](shape, target_area[condition_area](len(shape), a_mean))
                gen_location = lambda shape, size, x_max, y_max: convex_hull_controlled_random_location(shape, size, x_max, y_max, ch_d, k, g, min_move, stop, target_convexhull[condition_location](len(shape), ch_mean), tolerance)
                file_names = [f'{data_path}/{condition}/{params}/{m}.npy' for m in metric_id]
                if np.any([not os.path.exists(f) for f in file_names]):
                    os.makedirs(f'{data_path}/{condition}/{params}', exist_ok=True)
                    d = gen_all(f'{data_path}/{condition}/{params}', n_sample, ns, img_dim, gen_size, gen_location, gen_shape, gen_color, n_try, metric)
                    for j,m in enumerate(metric_id):                      
                        np.save(file_names[j], d[m])
                else:
                    d = {}
                    for j,m in enumerate(metric_id):                        
                        d[m] = np.load(file_names[j])
                file_names = [f'{data_path}/{condition}/{params}/{m}.npy' for m in more_metric_id]
                if np.any([not os.path.exists(f) for f in file_names]):
                    for j,m in enumerate(more_metric_id):
                        d[m] = more_metric[m](d)                     
                        np.save(file_names[j], d[m])
                else:
                    for j,m in enumerate(more_metric_id):                        
                        d[m] = np.load(file_names[j])
                for m in metric_id+more_metric_id:                        
                    data[condition][m].append(d[m])
                f = plot_statistics(ns, d, metric_name, {m: metric_target[m](ns, a_mean, ch_mean, condition_area, condition_location) for m in metric_id+more_metric_id}, metric_max)
                f.savefig(f'{figure_path}/{condition}_{params}.png')
                plt.close(f)

# -------------------------------
# Printing statistics
# -------------------------------

def print_cor(data, ns):
    for i in range(3):
        c = {m: np.corrcoef(data[m][i:].flatten(), ns[i:].flatten()) for m in metric_cor}
        s = f'For num>={i+1}:'
        for m in metric_cor[:-1]:
            s += f'C({m},num)={c[m][0,1]:.5f}, '
        m = metric_cor[-1]
        s += f'C({m},num)={c[m][0,1]:5f}'
        print(s)

def print_mean_std(data, f):
    mean_data = {m: np.mean(d) for m,d in data.items()}
    std_data = {m: np.std(d) for m,d in data.items()}
    for m,d in data.items():
        print(f'{m}: {mean_data[m]:{f[m]}}±{std_data[m]:{f[m]}}')

print("-------------------------------")
ns = np.tile(ns[:,None], n_sample*n_parameters)
for condition_size in size_method.keys():
    for condition_area in target_area.keys():
        for condition_location in target_convexhull.keys():
            condition = f'{condition_size}_{condition_area}_{condition_location}'
            data[condition] = {m: np.concatenate(data[condition][m], axis = 1) for m in metric_id+more_metric_id}
            print(condition)
            print_cor(data[condition], ns)
            print_mean_std(data[condition], metric_f)
            print("-------------------------------")

data = {m: np.concatenate([data[condition][m] for condition in data.keys()], axis = 1) for m in metric_id+more_metric_id}
m = len(size_method)*len(target_area)*len(target_convexhull)
ns = np.tile(ns, m)
print("All conditions combined")
print_cor(data, ns)
print_mean_std(data, metric_f)
print("-------------------------------")