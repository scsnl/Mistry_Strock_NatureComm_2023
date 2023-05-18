import os
import glob
from pathlib import PurePath
import numpy as np
import torch
import pytorch_lightning as pl
from torchvision import transforms
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from tqdm import tqdm
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import sys
import re

# -------------------------------
# Color generation choices
# -------------------------------

def fixed_color(n, bg_color, shape_color):
    return bg_color, [(shape_color,shape_color)]*n

def random_color(n, color_d):
    bg_color = np.random.randint(0,256,3)
    shape_color = np.random.randint(0,256,3)
    d = np.linalg.norm(shape_color - bg_color)
    while d < color_d:
        shape_color = np.random.randint(0,256,3)
        d = np.linalg.norm(shape_color - bg_color)
    bg_color = tuple(bg_color)
    shape_color = tuple(shape_color)
    return bg_color, [(shape_color,shape_color)]*n

# -------------------------------
# Shape generation choices
# -------------------------------

def circle_shape(n):
    return ["c"]*n

def square_shape(n):
    return ["s"]*n

def regular_polygon_shape(n, n_side):
    return [f'p{n_side}']*n

# -------------------------------
# Size generation choices
# -------------------------------

def area_controlled_fixed_size(shape, area):
    area_one = area/len(shape)
    return [area_to_size(s, area_one) for s in shape]

def area_controlled_random_size(shape, area, min_area_one, max_area_one):
    n = len(shape)
    size = []
    cumul_area = 0
    areas = []
    for i in range(n-1):
        min_area = np.max([area - cumul_area - (n-i-1)*max_area_one, min_area_one])
        max_area = np.min([area - cumul_area - (n-i-1)*min_area_one, max_area_one])
        if max_area < min_area:
            raise RuntimeError("Check your area_one conditions")
        area_one = np.random.uniform(min_area,max_area)
        cumul_area += area_one
        areas.append(area_one)
        size.append(area_to_size(shape[i], area_one))
    area_one = area-cumul_area
    size.append(area_to_size(shape[n-1], area_one))
    return size

# -------------------------------
# Location generation choices
# -------------------------------

def random_shape_location(shape, size, x_max, y_max, d):
    if shape == "c" or shape == "s":
        r = size
        x = np.random.uniform(r+d,x_max-r-d)
        y = np.random.uniform(r+d,y_max-r-d)
        return x,y
    elif shape[0] == "p":
        r = size
        x = np.random.uniform(r+d,x_max-r-d)
        y = np.random.uniform(r+d,y_max-r-d)
        theta = np.random.uniform(0,360)
        return x,y,theta
    else:
        raise RuntimeError(f'Shape {shape} does not exist')

def enclosing_circle(location, shape, size):
    if shape == "c":
        r = size
        x,y = location
    elif shape == "s":
        r = np.sqrt(2)*size
        x,y = location
    elif shape[0] == "p":
        r = size            
        x,y,_ = location
        r = size
    else:
        raise RuntimeError(f'Shape {shape} does not exist')
    return (x,y,r)

def no_constraint_random_location(shape, size, x_max, y_max, d, k, min_move, stop):
    #debug_path = f'{os.environ.get("DEBUG_PATH")}/numerosity/stimuli'
    #os.makedirs(f'{debug_path}', exist_ok=True)
    n = len(shape)
    i = 0
    location = np.array([random_shape_location(sh, sz, x_max, y_max, d) for sh,sz in zip(shape, size)])
    #color = fixed_color(n, (255,0,0), (0,0,0))
    #img = img_shape((224,224), shape, size, location, color)
    #img.save(f'{debug_path}/{n}_0_{i}.png')
    r = np.array([enclosing_circle(l,sh, sz)[2] for l,sh,sz in zip(location,shape,size)])
    min_distance = r[None, :]+r[:, None]+d
    direction = location[:,None,:2]-location[None,:,:2]
    distance = np.sqrt(direction[:,:,0]**2+direction[:,:,1]**2)
    idx = distance>0
    direction[idx,:] = direction[idx,:]/distance[idx,None]
    delta = distance - min_distance
    intersect = delta < 0
    move = np.zeros((n,2))
    max_xy = np.array([x_max-d, y_max-d]).reshape(1,2)
    min_xy = np.array([d, d]).reshape(1,2)
    while np.sum(intersect)>n:
        i+=1
        if i > stop:
            raise RuntimeError("Check your area conditions")
        delta[np.logical_not(intersect)] = 0
        move = -k*np.sum(direction*delta[:,:, None], axis = 1)
        idx2 = move>0
        max_right_move = max_xy-location[:,:2]-r[:,None]
        move[idx2] = np.clip(move[idx2], min_move, max_right_move[idx2])
        max_left_move = min_xy-location[:,:2]+r[:,None]
        idx3 = move<0
        move[idx3] = -np.clip(-move[idx3], min_move, -max_left_move[idx3])
        new_location = location[:,:2] + move
        location[:,:2] = location[:,:2] + move
        #img = img_shape((224,224), shape, size, location, color)
        #img.save(f'{debug_path}/{n}_0_{i}.png')
        direction = location[:,None,:2]-location[None,:,:2]
        distance = np.sqrt(direction[:,:,0]**2+direction[:,:,1]**2)
        idx = distance>0
        direction[idx,:] = direction[idx,:]/distance[idx,None]
        delta = distance - min_distance
        intersect = delta < 0
    return location, True

def convex_hull_controlled_random_location(shape, size, x_max, y_max, d, k, g, min_move, stop, convexhull, tolerance):
    #debug_path = f'{os.environ.get("DEBUG_PATH")}/numerosity/stimuli'
    #os.makedirs(f'{debug_path}', exist_ok=True)
    convexhullmin = convexhull - tolerance*convexhull
    convexhullmax = convexhull + tolerance*convexhull
    n = len(shape)
    i = 0
    location, _ =  no_constraint_random_location(shape, size, x_max, y_max, d, k, min_move, stop)
    #color = fixed_color(n, (255,0,0), (0,0,0))
    #img = img_shape((224,224), shape, size, location, color)
    #img.save(f'{debug_path}/{n}_1_{i}.png')
    r = np.array([enclosing_circle(l,sh, sz)[2] for l,sh,sz in zip(location,shape,size)])
    a = np.pi*r**2
    a_total = np.sum(a)
    min_distance = r[None, :]+r[:, None]+d
    direction = location[:,None,:2]-location[None,:,:2]
    distance = np.sqrt(direction[:,:,0]**2+direction[:,:,1]**2)
    idx = distance>0
    direction[idx,:] = direction[idx,:]/distance[idx,None]
    delta = distance - min_distance
    intersect = delta < 0
    move = np.zeros((n,2))
    max_xy = np.array([x_max-d, y_max-d]).reshape(1,2)
    min_xy = np.array([d, d]).reshape(1,2)
    current_convexhull = convex_hull_shape(shape, size, location)
    fixed = 0
    while i<=stop and fixed <= 5 and (np.sum(intersect)>n or current_convexhull < convexhullmin or current_convexhull > convexhullmax):
        i+=1
        mean_location = (np.sum(location[:,:2]*a[:,None], axis = 0)/a_total)[None, :]
        error = ((convexhull-current_convexhull)/convexhull)
        direction2 = error*(mean_location-location)
        delta[np.logical_not(intersect)] = 0
        p = np.clip(np.abs(error), 0, 1)
        p = 0.5
        mask = np.random.choice([0,1], p = [1-p,p], size = direction2.shape) # Trying to avoid oscillatory behavior
        mask = np.ones_like(direction2.shape)
        move = -g*mask*direction2-k*np.sum(direction*delta[:,:, None], axis = 1)
        idx2 = move>0
        idx3 = move<0
        max_right_move = np.clip(max_xy-location[:,:2]-r[:,None],0,None)
        move[idx2] = np.clip(move[idx2], min_move, max_right_move[idx2])
        max_left_move = np.clip(min_xy-location[:,:2]+r[:,None],None,0)
        move[idx3] = -np.clip(-move[idx3], min_move, -max_left_move[idx3])
        idx2 = move>0
        idx3 = move<0
        if np.sum(idx2)+np.sum(idx3) == 0:
            fixed += 1
        else:
            fixed = 0
        new_location = location[:,:2] + move
        location[:,:2] = location[:,:2] + move
        #img = img_shape((224,224), shape, size, location, color)
        #img.save(f'{debug_path}/{n}_1_{i}.png')
        direction = location[:,None,:2]-location[None,:,:2]
        distance = np.sqrt(direction[:,:,0]**2+direction[:,:,1]**2)
        idx = distance>0
        direction[idx,:] = direction[idx,:]/distance[idx,None]
        delta = distance - min_distance
        intersect = delta < 0
        current_convexhull = convex_hull_shape(shape, size, location)
    if i > stop or fixed > 5:
        m = f'Warning: Could not fit {n} dots of total area {a_total} in a convex hull {convexhull} ± {tolerance}% in {i} iterations'
        if fixed > 5:
            m = f'{m} (Stopped moving)'
        #print(m)
        return location, False
    else:
        return location, True
    
# -------------------------------
# Area calculation
# -------------------------------

def size_to_area(shape, size):
    if shape == "c":
        return np.pi*size**2
    elif shape == "s": 
        return 4*size**2
    elif shape[0] == "p":
        n = int(shape[1:])
        return (n*np.sin(2*np.pi/n)/2)*size**2
    else:
        raise RuntimeError(f'Shape {shape} does not exist')

def area_to_size(shape, area):
    if shape == "c":
        return np.sqrt(area/np.pi)
    elif shape == "s": 
        return np.sqrt(area/4)
    elif shape[0] == "p":
        n = int(shape[1:])
        return np.sqrt(area/(n*np.sin(2*np.pi/n)/2))
    else:
        raise RuntimeError(f'Shape {shape} does not exist')

def area_shape(shape, size):
    area = [size_to_area(sh,sz) for sh,sz in zip(shape, size)]
    return np.sum(area)

no_transform = transforms.ToTensor()

def area_img(img, bg_color):
    bg_color = no_transform(np.array(bg_color, dtype = np.uint8).reshape((1,1,3)))
    tensor = no_transform(img)
    idx = torch.sum(tensor != bg_color, axis = 0)>0
    return torch.sum(idx)

# -------------------------------
# Perimeter calculation
# -------------------------------

def size_to_perimeter(shape, size):
    if shape == "c":
        return 2*np.pi*size
    elif shape == "s": 
        return 8*size
    elif shape[0] == "p":
        n = int(shape[1:])
        return 2*n*np.sin(np.pi/n)*size
    else:
        raise RuntimeError(f'Shape {shape} does not exist')

def perimeter_to_size(shape, perimeter):
    if shape == "c":
        return perimeter/(2*np.pi)
    elif shape == "s": 
        return perimeter/8
    elif shape[0] == "p":
        n = int(shape[1:])
        return perimeter/(2*n*np.sin(np.pi/n))
    else:
        raise RuntimeError(f'Shape {shape} does not exist')

def perimeter_shape(shape, size):
    perimeter = [size_to_perimeter(sh,sz) for sh,sz in zip(shape, size)]
    return np.sum(perimeter)

# -------------------------------
# Convex hull calculation
# -------------------------------

def point_shape(shape, size, location):
    if shape == "c":
        x,y = location
        r = size
        corners = [x-r,y-r,x+r-1,y+r-1]
        x,y,r = x-0.5,y-0.5,r-0.5
        n = 100
        k = np.arange(0,n)
        angle = 2*k*np.pi/n
        point = np.array([[x+r*np.cos(a),y+r*np.sin(a)] for a in angle])
    elif shape == "s":
        x,y = location
        r = size
        x0,x1 = x-r,x+r
        y0,y1 = y-r,y+r
        point = np.array([[x0,y0],[x0,y1],[x1,y0],[x1,y1]])
    elif shape[0] == "p": 
        n = int(shape[1:])
        x,y,theta = location
        r = size
        x,y,r = x-0.5,y-0.5,r-0.5
        k = np.arange(0,n)
        angle = (theta + 2*k*np.pi/n)
        point = np.array([[x+r*np.cos(a),y+r*np.sin(a)] for a in angle])
    else:
        raise RuntimeError(f'Shape {shape} does not exist')
    return point

def convex_hull_shape(shape, size, location):
    n = len(shape)
    point = []
    for i in range(n):
        point.append(point_shape(shape[i], size[i], location[i]))
    point = np.concatenate(point)
    hull = ConvexHull(point)
    return hull.volume

def convex_hull_img(img, bg_color):
    bg_color = no_transform(np.array(bg_color, dtype = np.uint8).reshape((1,1,3)))
    tensor = no_transform(img)
    x,y = np.where((torch.sum(tensor != bg_color, axis = 0)>0).numpy())
    n = len(x)
    point = []
    for i in range(n):
        point.append(point_shape('s', 0.5, (x[i]+0.5,y[i]+0.5)))
    point = np.concatenate(point)
    hull = ConvexHull(point)
    return hull.volume


# -------------------------------
# Core generations of img
# -------------------------------

def draw_shape(draw, shape, size, location, color):
    fill, outline = color
    if shape == "c":
        x,y = location
        r = size
        corners = [x-r,y-r,x+r-1,y+r-1]
        draw.ellipse(corners, fill = fill, outline = outline)
    elif shape == "s":
        x,y = location
        r = size
        corners = [x-r,y-r,x+r-1,y+r-1]
        draw.rectangle(corners, fill = fill, outline = outline)
    elif shape[0] == "p": # TODO: Find a draw that approximate better the area
        n = int(shape[1:])
        x,y,theta = location
        r = size
        #x,y,r = x-0.5,y-0.5,r-0.5
        #k = np.arange(0,n)
        #angle = (theta + 2*k*np.pi/n)
        #vertex = [(x+r*np.cos(a),y+r*np.sin(a)) for a in angle]
        #draw.polygon(vertex, fill = fill, outline = outline)
        draw.regular_polygon(((x-0.5,y-0.5),r-0.5), n, rotation = theta, fill = fill, outline = outline)
    else:
        raise RuntimeError(f'Shape {shape} does not exist')

def img_shape(img_dim, shape, size, location, color):
    n = len(shape)
    bg_color, shape_color = color
    img = Image.new('RGB', img_dim, color = bg_color)
    draw = ImageDraw.Draw(img)
    for i in range(n):
        draw_shape(draw, shape[i], size[i], location[i], shape_color[i])
    return img

def gen_shape_img(n, img_dim, gen_size, gen_location, gen_shape, gen_color, n_try, metric):
    success = False
    k = 0
    while not success and k < n_try:
        k+=1
        color = gen_color(n)
        shape = gen_shape(n)
        size = gen_size(shape)
        location, success = gen_location(shape, size, *img_dim)
    img = img_shape(img_dim, shape, size, location, color)
    data = {}
    for m in metric.keys():
        data[m] = metric[m](color, shape, size, location, success, img)
    return img, data

def gen_all(path, n_sample, ns, img_dim, gen_size, gen_location, gen_shape, gen_color, n_try, metric):
    n_number = len(ns)
    data = {}
    for m in metric.keys():
        data[m] = np.empty((n_number, n_sample))
    for i,n in enumerate(ns):
        for j in range(n_sample):
            img, d = gen_shape_img(n, img_dim, gen_size, gen_location, gen_shape, gen_color, n_try, metric)
            for m in metric.keys():
                data[m][i,j] = d[m]
            img.save(f'{path}/{n}_{j}.png')
    return data

# -------------------------------
# Interface with pytorch
# -------------------------------

# Normalization used to train on ImageNet
default_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

class CountingDotsDataSet(Dataset):

    def __init__(self, path, include = None, transform = default_transform):
        self.transform = transform
        self.path = sorted(glob.glob(path))
        if not include is None:
            self.path = sorted([p for p in self.path if re.fullmatch(include, p)])
        self.all_num, self.label = np.unique([self.get_num_from_path(p) for p in self.path], return_inverse=True)
        self.all_param, self.param_id = np.unique([self.get_param_from_path(p) for p in self.path], return_inverse=True)
        self.all_condition, self.condition_id = np.unique([self.get_condition_from_path(p) for p in self.path], return_inverse=True)

    def __len__(self):
        return len(self.path)

    def get_num_from_path(self,path):
        return int(PurePath(path).parts[-1].split('_')[0])

    def get_id_from_path(self,path):
        return int(PurePath(path).parts[-1].split('.')[0].split('_')[1])

    def get_param_from_path(self,path):
        return PurePath(path).parts[-2]

    def get_condition_from_path(self,path):
        return PurePath(path).parts[-3]

    def properties(self, idx, properties = []):
        path = self.path[idx]
        label = self.label[idx]
        n = self.all_num[label]
        k = self.get_id_from_path(path)
        path =  PurePath(path).parent
        d = {'n': n}
        for p in properties:
            d[p] = np.load(f'{path}/{p}.npy')[label,k]
        return d

    def __getitem__(self, idx):
        img = self.transform(Image.open(self.path[idx]))
        label = self.label[idx]
        param_id = self.param_id[idx]
        condition_id = self.condition_id[idx]
        return  img,label,condition_id,param_id