# -*- coding: utf-8 -*-
"""
Created on Mon Sep 05 16:37:22 2016

Analyze Stamp labeled images.

@author: Babak, adapted by Guido
"""

import multiresolutionimageinterface as mir
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


def tissueComponentAmount(low_res_image, pix_area):
    amount_benign = np.sum(np.equal(low_res_image, 1))*pix_area
    amount_dcis = np.sum(np.equal(low_res_image, 2))*pix_area
    amount_idc = np.sum(np.equal(low_res_image, 3))*pix_area
    Total_Tissue = amount_benign + amount_dcis + amount_idc
    return amount_benign, amount_dcis, amount_idc, Total_Tissue
    
def separateBiopsies(low_res_image):
    TissueMask = np.array(low_res_image, copy=True)
    #TissueMask[TissueMask == 1] = 0
    TissueMask[TissueMask != 0] = 1
    #plt.imshow(TissueMask[:,:,0])
    TissueMask = TissueMask.astype(bool)
    TissueMask_dilated = binary_closing(TissueMask[:,:,0], disk(10))  
    #plt.imshow(TissueMask_dilated[:,:])
    labeled_tissue_mask = measure.label(TissueMask_dilated, connectivity = 2) 
    return labeled_tissue_mask
    
def tissueMorphology(low_res_image, labeled_tissue_mask, pix_area, data_level):
    idc_mask = np.array(low_res_image, copy=True)
    idc_mask[idc_mask != 3] = 0
    idc_mask = idc_mask/3
    idc_mask_holes_filled = binary_fill_holes(idc_mask[:,:,0]).astype(int)
    #plt.imshow(idc_mask[:,:,0])
    idc_Labeled = measure.label(idc_mask, connectivity = 2) 
    area_voronoi = watershed(labeled_tissue_mask, idc_Labeled[:,:,0])
    idc_labeled_holes_filled = measure.label(idc_mask_holes_filled, connectivity = 2) 
    #plt.imshow(area_voronoi)
    #plt.imsave('C:\Users\Babak\Documents\sth.tif', area_voronoi*(1-idc_mask[:,:,0]/3),  cmap='Greys')    
    #plt.imsave('C:\Users\Babak\Documents\sth2.tif', idc_mask_holes_filled,  cmap='Greys')    
    
    properties = measure.regionprops(idc_Labeled)
    properties_area_voronoi = measure.regionprops(area_voronoi)
    properties_hole_filled = measure.regionprops(idc_labeled_holes_filled)
    MaxLabel = np.amax(idc_Labeled)
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
        current_area = properties[i].area*pix_area
        if current_area > 100*100/(pow(2,data_level)*pow(2,data_level)):
            num_glands += 1
            area_all.append(current_area)
            eccentricity_all.append(properties[i].eccentricity)
            centroids_all.setdefault(labeled_tissue_mask[int(properties[i].centroid[0]), int(properties[i].centroid[1])],[]).append(properties[i].centroid)
            index_voronoi = area_voronoi[int(properties[i].coords[0,0]), int(properties[i].coords[0,1])]
            current_vor_area = properties_area_voronoi[index_voronoi-1].area*pix_area     
            
            index_idc_filled = idc_labeled_holes_filled[int(properties[i].coords[0,0]), int(properties[i].coords[0,1])]
            current_filled_area = properties_hole_filled[index_idc_filled-1].area*pix_area 
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
    
    
def fillAdjacencyAndDist(adjacency_matrix, distance_matrix, pixel_to_micrometer, indices, simplex, i, j, edge_threshold):
    if adjacency_matrix[indices[i], indices[j]] != 1: 
        dist = math.sqrt(pow(simplex[i,0]-simplex[j,0], 2) + pow(simplex[i,1]-simplex[j,1], 2)) * pixel_to_micrometer
        if dist<edge_threshold: # comparison of edge length is in micrometer s
            distance_matrix[indices[i], indices[j]] = dist
            distance_matrix[indices[j], indices[i]] = dist
            adjacency_matrix[indices[i], indices[j]] = 1
            adjacency_matrix[indices[j], indices[i]] = 1
    return adjacency_matrix, distance_matrix

def extractDelaunayFeatures(points_dict, pixel_spacing, data_level):
    num_neighbors_all, Average_node_dist = [], []
    number_of_biopsy_clusters = 0
    for groups in points_dict:
        points = np.asarray(points_dict[groups])
        if len(points)>2 and groups!=0:
            tri = Delaunay(points)
#            plt.triplot(points[:,1], points[:,0], tri.simplices.copy())
#            plt.plot(points[:,1], points[:,0], 'o')
#            plt.ylim(0, 6624)
#            plt.xlim(0, 12960)
#            plt.show()        
            adjacency_matrix = np.zeros((len(points),len(points)), dtype = bool)
            distance_matrix = np.zeros((len(points),len(points)), dtype = float)
            pixel_to_micrometer = pixel_spacing[0]*pow(2,data_level)
            edge_threshold = 5000 #micrometers
            for vertex in range(len(tri.simplices)):
                indices = tri.simplices[vertex,:]
                simplex = points[indices]
                adjacency_matrix, distance_matrix = fillAdjacencyAndDist(adjacency_matrix, distance_matrix, pixel_to_micrometer, indices, simplex, 0, 1, edge_threshold)
                adjacency_matrix, distance_matrix = fillAdjacencyAndDist(adjacency_matrix, distance_matrix, pixel_to_micrometer, indices, simplex, 0, 2, edge_threshold)
                adjacency_matrix, distance_matrix = fillAdjacencyAndDist(adjacency_matrix, distance_matrix, pixel_to_micrometer, indices, simplex, 1, 2, edge_threshold)
            
            num_neighbors = np.sum(adjacency_matrix, axis=0)
            num_neighbors_all.append(np.sum(adjacency_matrix, axis=0))
            Average_node_dist.append(np.divide(np.sum(distance_matrix, axis=0), num_neighbors))
            number_of_biopsy_clusters += 1

    if number_of_biopsy_clusters==0:
        number_of_biopsy_clusters = 1
    return number_of_biopsy_clusters, num_neighbors_all, Average_node_dist
    
if __name__ == "__main__":
    r = mir.MultiResolutionImageReader()
    r2 = mir.MultiResolutionImageReader()
    
    
    result_image_dir = r'C:\upload\TestSetResultsLastRound'
    image_path_all = r'G:\Dataset\Dataset_Harvard\babak\Images\Testing'
    outputCSV = r'C:\upload\TrainResultsLastRound\TestFeatures7sep.csv'
    image_file_list = []
    image_file_list += [os.path.join(image_path_all, each) for each in os.listdir(image_path_all) if (each.endswith('.svs') or each.endswith('.ndpi')) ] #or each.endswith('.ndpi')
    
    f = open(outputCSV, 'wt')
    csvwriter = csv.writer(f, delimiter=',', lineterminator='\n')
    
    data_level = 4
    start_val = 0
    for image_path in image_file_list:
        ip = image_path.rstrip('\n')
        base_name = os.path.splitext(ntpath.basename(ip))[0]    
        result_path = os.path.join(result_image_dir, base_name + '_result.tif')
        if os.path.isfile(result_path):
            print "Processing image: ", start_val, "/", len(image_file_list), "---", image_path
            im_image = r.open(image_path)   
            im_result = r2.open(result_path)  
            
            dims = im_result.getLevelDimensions(data_level)  
            pixel_spacing = im_image.getSpacing()
            pix_area = pixel_spacing[0]*pixel_spacing[1]
            low_res_image = im_result.getUCharPatch(0, 0, dims[0], dims[1], data_level)  
            amount_benign, amount_dcis, amount_idc, Total_Tissue = tissueComponentAmount(low_res_image, pix_area)
            
            labeled_tissue_mask = separateBiopsies(low_res_image)
            
            area_all, eccentricity_all, Voronoi_area, Voronoi_area_ZOI, filled_ratio_area, points_dict, num_glands = \
                tissueMorphology(low_res_image, labeled_tissue_mask, pix_area, data_level)
            
            number_of_biopsy_clusters, num_neighbors_all, Average_node_dist = extractDelaunayFeatures(points_dict, pixel_spacing, data_level)
                
            num_neighbors_all = np.array([item for sublist in num_neighbors_all for item in sublist])
            num_neighbors_all = num_neighbors_all[num_neighbors_all != 0]
            
            Average_node_dist = np.array([item for sublist in Average_node_dist for item in sublist])
            Average_node_dist = Average_node_dist[~np.isnan(Average_node_dist)]       
            
            area_all, eccentricity_all, Voronoi_area, Voronoi_area_ZOI, filled_ratio_area, num_neighbors_all, Average_node_dist = \
                handleXZerosInSTD(area_all, eccentricity_all, Voronoi_area, Voronoi_area_ZOI, filled_ratio_area, num_neighbors_all, Average_node_dist)        
            
            csvwriter.writerow( (base_name, amount_benign, amount_dcis, amount_idc, amount_benign/Total_Tissue, amount_dcis/Total_Tissue, amount_idc/Total_Tissue,\
                np.mean(area_all), np.median(area_all), np.std(area_all), np.percentile(area_all, 75) - np.percentile(area_all, 25), np.max(area_all),\
                np.mean(eccentricity_all), np.median(eccentricity_all), np.std(eccentricity_all), np.percentile(eccentricity_all, 75) - np.percentile(eccentricity_all, 25),\
                np.mean(Voronoi_area), np.median(Voronoi_area), np.std(Voronoi_area), np.percentile(Voronoi_area, 75) - np.percentile(Voronoi_area, 25),\
                np.mean(Voronoi_area_ZOI), np.median(Voronoi_area_ZOI), np.std(Voronoi_area_ZOI), np.percentile(Voronoi_area_ZOI, 75) - np.percentile(Voronoi_area_ZOI, 25),\
                np.mean(filled_ratio_area), np.median(filled_ratio_area), np.std(filled_ratio_area), np.percentile(filled_ratio_area, 75) - np.percentile(filled_ratio_area, 25),\
                np.mean(num_neighbors_all), np.median(num_neighbors_all), np.std(num_neighbors_all), np.percentile(num_neighbors_all, 75) - np.percentile(num_neighbors_all, 25),\
                np.mean(Average_node_dist), np.median(Average_node_dist), np.std(Average_node_dist), np.percentile(Average_node_dist, 75) - np.percentile(Average_node_dist, 25), \
                np.max(Average_node_dist), num_glands/float(number_of_biopsy_clusters) ))
                
                
            im_image.close()
            im_result.close()
            start_val+=1
    f.close()       
    
    
    
    
    
    
    
    
    