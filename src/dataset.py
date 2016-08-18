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
	benign = sorted(glob(os.path.join(P.DATA_FOLDER, 'Originals/Train/Label1','*.tif')))
	dcis = sorted(glob(os.path.join(P.DATA_FOLDER, 'Originals/Train/Label2','*.tif')))
	idc = sorted(glob(os.path.join(P.DATA_FOLDER, 'Originals/Train/Label3','*.tif')))

	if shuffle:
		 map(np.random.shuffle, [benign, dcis, idc])

	return benign, dcis, idc

def validation_filenames(shuffle=True):
	benign = sorted(glob(os.path.join(P.DATA_FOLDER, 'Originals/Validation/Label1','*.tif')))
	dcis = sorted(glob(os.path.join(P.DATA_FOLDER, 'Originals/Validation/Label2','*.tif')))
	idc = sorted(glob(os.path.join(P.DATA_FOLDER, 'Originals/Validation/Label3','*.tif')))

	if shuffle:
		 map(np.random.shuffle, [benign, dcis, idc])

	return benign, dcis, idc

def mask_folder():
	if 'L1' in P.DATA_FOLDER:
		path = os.path.join(P.DATA_FOLDER, 'Masks/MasksAllBinary')
	else:
		path = os.path.join(P.DATA_FOLDER, 'Masks/MasksAll2class')
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
