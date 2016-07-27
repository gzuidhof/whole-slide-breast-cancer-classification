import numpy as np
from glob import glob
import os
import random
import ntpath
from params import params as P

label_names = {
	0: 'Benign',
	1: 'DCIS',
	2: 'IDC'
}



def train_filenames(shuffle=True):
	benign = sorted(glob(os.path.join(P.DATA_FOLDER, 'Benign_Train','*.tif')))
	dcis = sorted(glob(os.path.join(P.DATA_FOLDER, 'DCIS_Train','*.tif')))
	idc = sorted(glob(os.path.join(P.DATA_FOLDER, 'IDC_train','*.tif')))

	if shuffle:
		 map(np.random.shuffle, [benign, dcis, idc])

	return benign, dcis, idc

def validation_filenames(shuffle=True):
	benign = sorted(glob(os.path.join(P.DATA_FOLDER, 'Benign_Validation','*.tif')))
	dcis = sorted(glob(os.path.join(P.DATA_FOLDER, 'DCIS_Validation','*.tif')))
	idc = sorted(glob(os.path.join(P.DATA_FOLDER, 'IDC_validation','*.tif')))

	if shuffle:
		 map(np.random.shuffle, [benign, dcis, idc])

	return benign, dcis, idc

def mask_folder():
	path = os.path.join(P.DATA_FOLDER, 'AllMasksMerged')
	return path

def per_class_filelist(Benign_file_list, DCIS_file_list, IDC_file_list, msk_fls_All, msk_src, num_each_class):
	random_Samples = []
	random_Samples = (Benign_file_list[:num_each_class[0]] +
		DCIS_file_list[:num_each_class[1]] +
		IDC_file_list[:num_each_class[2]] )

	# get rid of \m at the end of the list elements
	random_Samples = map(lambda s: s.strip(), random_Samples)

	msk_fls = []
	for ip in random_Samples:
		ip = ip.rstrip('\n')
		base_name = os.path.splitext(ntpath.basename(ip))[0]
		msk_fls.append(os.path.join(msk_fls_All, base_name +'_Mask.tif'))

	for i in range(len(random_Samples)):
		msk_src[random_Samples[i]] = msk_fls[i]

	return random_Samples, msk_src

def label_name(label):
	try:
		return label_names[label]
	except:
		return label_names[np.argmax(label)]
