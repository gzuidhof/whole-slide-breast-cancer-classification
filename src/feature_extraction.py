# -*- coding: utf-8 -*-
"""
Created on Mon Sep 05 16:37:22 2016

Analyze Stamp labeled images.

@author: Babak, adapted by Guido
"""
from __future__ import division
import os
import ntpath
import math
from skimage.transform import resize
#import scipy.ndimage
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from scipy.spatial import Delaunay
from skimage.morphology import binary_closing
from skimage.morphology import disk
from skimage.morphology import watershed
from scipy.ndimage.morphology import binary_fill_holes
import csv
from glob import glob

def tissueComponentAmount(low_res_image):
    amount_benign = np.sum(np.equal(low_res_image, 1))#*pix_area
    amount_dcis = np.sum(np.equal(low_res_image, 2))#*pix_area
    amount_idc = np.sum(np.equal(low_res_image, 3))#*pix_area
    amount_cancer = amount_dcis + amount_idc
    Total_Tissue = amount_benign + amount_dcis + amount_idc


    #print amount_benign, amount_dcis, amount_idc, Total_Tissue
    #exit()

    return amount_benign, amount_dcis, amount_idc, amount_cancer, Total_Tissue
    
def separateBiopsies(low_res_image):
    TissueMask = np.array(low_res_image, copy=True)
    #TissueMask[TissueMask == 1] = 0
    TissueMask[TissueMask != 0] = 1
    #plt.imshow(TissueMask[:,:,0])
    #plt.show()
    TissueMask = TissueMask.astype(bool)
    #print TissueMask.shape
    TissueMask_dilated = binary_closing(TissueMask, disk(2))  
    #plt.imshow(TissueMask_dilated[:,:])
    #plt.show()
    labeled_tissue_mask = measure.label(TissueMask_dilated, connectivity = 2) 
    return labeled_tissue_mask
    
def tissueMorphology(low_res_image, labeled_tissue_mask, class_pixel_value): #pix_area):
    class_mask = np.array(low_res_image, copy=True)
    class_mask[class_mask != class_pixel_value] = 0
    class_mask = class_mask/class_pixel_value
    class_mask_holes_filled = binary_fill_holes(class_mask).astype(int)
    #plt.imshow(class_mask[:,:])
    class_Labeled = measure.label(class_mask, connectivity = 2) 
    area_voronoi = watershed(labeled_tissue_mask, class_Labeled)
    class_labeled_holes_filled = measure.label(class_mask_holes_filled, connectivity = 2) 
    im = np.hstack((area_voronoi/np.max(area_voronoi), low_res_image))
    #plt.imshow(im); plt.show()
    #plt.imsave('C:\Users\Babak\Documents\sth.tif', area_voronoi*(1-class_mask[:,:,0]/3),  cmap='Greys')    
    #plt.imsave('C:\Users\Babak\Documents\sth2.tif', class_mask_holes_filled,  cmap='Greys')    
    
    properties = measure.regionprops(class_Labeled)
    properties_area_voronoi = measure.regionprops(area_voronoi)
    properties_hole_filled = measure.regionprops(class_labeled_holes_filled)
    MaxLabel = np.amax(class_Labeled)
    num_glands = 0
    area_all = []
    eccentricity_all = []
    centroids_all = {}
    Voronoi_area = []
    Voronoi_area_ZOI = []
    filled_ratio_area = []
    #index_raw_area = []
    #index_watershed = []
    for i in range(0, MaxLabel):
        current_area = properties[i].area#*pix_area
        #print current_area
        if current_area > 3:#100*100/(pow(2,data_level)*pow(2,data_level)):
            num_glands += 1
            area_all.append(current_area)
            eccentricity_all.append(properties[i].eccentricity)
            centroids_all.setdefault(labeled_tissue_mask[int(properties[i].centroid[0]), int(properties[i].centroid[1])],[]).append(properties[i].centroid)
            index_voronoi = area_voronoi[int(properties[i].coords[0,0]), int(properties[i].coords[0,1])]
            current_vor_area = properties_area_voronoi[index_voronoi-1].area#*pix_area     
            
            index_class_filled = class_labeled_holes_filled[int(properties[i].coords[0,0]), int(properties[i].coords[0,1])]
            current_filled_area = properties_hole_filled[index_class_filled-1].area#*pix_area 
            Voronoi_area.append(current_vor_area)
            Voronoi_area_ZOI.append(current_area/current_vor_area)
            filled_ratio_area.append(current_area/current_filled_area)
    return area_all, eccentricity_all, Voronoi_area, Voronoi_area_ZOI, filled_ratio_area, centroids_all, num_glands

def handleXZerosInSTD(area_all, eccentricity_all, Voronoi_area, Voronoi_area_ZOI, filled_ratio_area, num_neighbors_all, Average_node_dist):
    if len(area_all)==0:
        area_all = [0]
    if len(eccentricity_all)==0:
        eccentricity_all = [0]
    if len(Voronoi_area)==0:
        Voronoi_area = [0]        
    if len(Voronoi_area_ZOI)==0:
        Voronoi_area_ZOI = [0]
    if len(filled_ratio_area)==0:
        filled_ratio_area = [0]        
    if len(num_neighbors_all)==0:
        num_neighbors_all = [0]        
    if len(Average_node_dist)==0:
        Average_node_dist = [0]             
    return area_all, eccentricity_all, Voronoi_area, Voronoi_area_ZOI, filled_ratio_area, num_neighbors_all, Average_node_dist
    
    
def fillAdjacencyAndDist(adjacency_matrix, distance_matrix, indices, simplex, i, j, edge_threshold):
    if adjacency_matrix[indices[i], indices[j]] != 1: 
        dist = math.sqrt(pow(simplex[i,0]-simplex[j,0], 2) + pow(simplex[i,1]-simplex[j,1], 2))# * pixel_to_micrometer

        if dist<edge_threshold: # comparison of edge length in pixels
            distance_matrix[indices[i], indices[j]] = dist
            distance_matrix[indices[j], indices[i]] = dist
            adjacency_matrix[indices[i], indices[j]] = 1
            adjacency_matrix[indices[j], indices[i]] = 1
    return adjacency_matrix, distance_matrix

def extractDelaunayFeatures(points_dict, image):
    num_neighbors_all, Average_node_dist = [], []
    number_of_biopsy_clusters = 0
    for groups in points_dict:
        points = np.asarray(points_dict[groups])
        if len(points)>2 and groups!=0:
            tri = Delaunay(points)
            #plt.imshow(image)
            #plt.triplot(points[:,1], points[:,0], tri.simplices.copy())
            #plt.plot(points[:,1], points[:,0], 'o')
            #plt.ylim(0, image.shape[0])
            #plt.xlim(0, image.shape[1])
            #plt.show()        
            adjacency_matrix = np.zeros((len(points),len(points)), dtype = bool)
            distance_matrix = np.zeros((len(points),len(points)), dtype = float)
            edge_threshold = 24 #pixels
            for vertex in range(len(tri.simplices)):
                indices = tri.simplices[vertex,:]
                simplex = points[indices]
                adjacency_matrix, distance_matrix = fillAdjacencyAndDist(adjacency_matrix, distance_matrix, indices, simplex, 0, 1, edge_threshold)
                adjacency_matrix, distance_matrix = fillAdjacencyAndDist(adjacency_matrix, distance_matrix, indices, simplex, 0, 2, edge_threshold)
                adjacency_matrix, distance_matrix = fillAdjacencyAndDist(adjacency_matrix, distance_matrix, indices, simplex, 1, 2, edge_threshold)
            
            num_neighbors = np.sum(adjacency_matrix, axis=0)
            num_neighbors_all.append(np.sum(adjacency_matrix, axis=0))
            Average_node_dist.append(np.divide(np.sum(distance_matrix, axis=0), num_neighbors))
            number_of_biopsy_clusters += 1

    if number_of_biopsy_clusters==0:
        number_of_biopsy_clusters = 1
    return number_of_biopsy_clusters, num_neighbors_all, Average_node_dist

def load_label_file(path):
    with open(path) as f:
        lines = f.readlines()

    cur_label = -1
    cur_set = -1

    labels = {}
    sets = {}

    for l in lines:
        if '#' in l:
            if 'train' in l:
                cur_set = 0
            elif 'val' in l:
                cur_set = 1
            elif 'test' in l:
                cur_set = 2
            else:
                print "UNKNOWN SET in line ", l
                assert False
            
            if 'benign' in l:
                cur_label = 0
            elif 'dcis' in l:
                cur_label = 1
            elif 'idc' in l:
                cur_label = 2
            else:
                print "UNKNOWN LABEL in line ", l
                assert False

        else:
            slide_name = l.replace('\n',"")
            labels[slide_name] = cur_label
            sets[slide_name] = cur_set

    return labels, sets

if __name__ == "__main__":
    result_image_dir = r'../extracted_features'

    if not os.path.exists(result_image_dir):
        os.makedirs(result_image_dir)

    image_path_all = r'../wsi_predictions/'
    label_file_path = os.path.join(image_path_all, 'slides.txt')

    labels, sets = load_label_file(label_file_path)


    outputCSV = r'../extracted_features/features.csv'
    image_file_list = glob(image_path_all + '*.npy')

    f = open(outputCSV, 'wt')
    csvwriter = csv.writer(f, delimiter=',', lineterminator='\n')
    
    for c, image_path in enumerate(image_file_list):
        ip = image_path.rstrip('\n')
        base_name = os.path.splitext(ntpath.basename(ip))[0]    
        #result_path = os.path.join(result_image_dir, base_name + '_result.png')
        
        print "Processing image: ", c+1, "/", len(image_file_list), "---", image_path

        image = np.load(image_path)
        background = np.where(np.sum(image, axis=2) == 0)

        image = np.argmax(image, axis=2)+1
        image[background] = 0
        #plt.imshow(image); plt.show()

        DCIS_CLASS = 2
        IDC_CLASS = 3
        
        amount_benign, amount_dcis, amount_idc, amount_cancer, Total_Tissue = tissueComponentAmount(image)
        
        labeled_tissue_mask = separateBiopsies(image)
        



        def get_descriptors(value):
            return [np.mean(value), np.median(value), np.std(value), np.percentile(value, 75) - np.percentile(value, 25)]


        def features_for_class(_CLASS):

            area_all, eccentricity_all, Voronoi_area, Voronoi_area_ZOI, filled_ratio_area, points_dict, num_glands = \
                tissueMorphology(image, labeled_tissue_mask, _CLASS)
            
            number_of_biopsy_clusters, num_neighbors_all, Average_node_dist = extractDelaunayFeatures(points_dict, image)
                
            num_neighbors_all = np.array([item for sublist in num_neighbors_all for item in sublist])
            num_neighbors_all = num_neighbors_all[num_neighbors_all != 0]
            
            Average_node_dist = np.array([item for sublist in Average_node_dist for item in sublist])
            Average_node_dist = Average_node_dist[~np.isnan(Average_node_dist)]       
            
            area_all, eccentricity_all, Voronoi_area, Voronoi_area_ZOI, filled_ratio_area, num_neighbors_all, Average_node_dist = \
                handleXZerosInSTD(area_all, eccentricity_all, Voronoi_area, Voronoi_area_ZOI, filled_ratio_area, num_neighbors_all, Average_node_dist)        
        

            f_area = get_descriptors(area_all)
            f_eccentricity = get_descriptors(eccentricity_all)
            f_voronoi_area = get_descriptors(Voronoi_area)
            f_voronoi_area_ZOI = get_descriptors(Voronoi_area_ZOI)
            f_filled_ratio_area = get_descriptors(filled_ratio_area)
            f_num_neighbors = get_descriptors(num_neighbors_all)
            f_average_node_dist = get_descriptors(Average_node_dist)

            f_max_area = np.max(area_all)
            f_max_avg_node_dist = np.max(Average_node_dist)
            f_num_clusters = num_glands/float(number_of_biopsy_clusters)



            feature_set = f_area + f_eccentricity + f_voronoi_area + f_voronoi_area_ZOI + f_filled_ratio_area + f_num_neighbors + f_average_node_dist + [f_max_area + f_max_avg_node_dist + f_num_clusters]

            return feature_set

        f_dcis = features_for_class(DCIS_CLASS)
        f_idc = features_for_class(IDC_CLASS)



        slide_name = "_".join(base_name.split('_')[:-1])

        label = labels[slide_name]
        subset = sets[slide_name] #train, validation or test

        features =  [base_name, label, subset, amount_benign, amount_dcis, amount_idc, amount_cancer, \
            amount_benign/Total_Tissue, amount_dcis/Total_Tissue, amount_idc/Total_Tissue, amount_cancer/Total_Tissue, amount_dcis/amount_cancer, amount_idc/amount_cancer] + \
            f_idc + f_dcis

        if c == 0:
            header = ['name','label','subset'] + ['f'+str(i) for i in xrange(len(features)-3)]
            csvwriter.writerow(header)
        
        

        #print features
        csvwriter.writerow( features)

    f.close()       
    
    
    
    
    
    
    
    
    