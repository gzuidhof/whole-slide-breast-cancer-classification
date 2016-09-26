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
	if 'Aug17' in P.DATA_FOLDER:
		benign = sorted(glob(os.path.join(P.DATA_FOLDER, 'Original/Train/Label1','*.jpeg')))
		dcis = sorted(glob(os.path.join(P.DATA_FOLDER, 'Original/Train/Label2','*.jpeg')))
		idc = sorted(glob(os.path.join(P.DATA_FOLDER, 'Original/Train/Label3','*.jpeg')))
	else:
		benign = sorted(glob(os.path.join(P.DATA_FOLDER, 'Originals/Train/Label1','*.tif')))
		dcis = sorted(glob(os.path.join(P.DATA_FOLDER, 'Originals/Train/Label2','*.tif')))
		idc = sorted(glob(os.path.join(P.DATA_FOLDER, 'Originals/Train/Label3','*.tif')))

	if shuffle:
		 map(np.random.shuffle, [benign, dcis, idc])

	return benign, dcis, idc

def validation_filenames(shuffle=True):

	if 'Aug17' in P.DATA_FOLDER:
		benign = sorted(glob(os.path.join(P.DATA_FOLDER, 'Original/Validation/Label1','*.jpeg')))
		dcis = sorted(glob(os.path.join(P.DATA_FOLDER, 'Original/Validation/Label2','*.jpeg')))
		idc = sorted(glob(os.path.join(P.DATA_FOLDER, 'Original/Validation/Label3','*.jpeg')))
	else:
		benign = sorted(glob(os.path.join(P.DATA_FOLDER, 'Originals/Validation/Label1','*.tif')))
		dcis = sorted(glob(os.path.join(P.DATA_FOLDER, 'Originals/Validation/Label2','*.tif')))
		idc = sorted(glob(os.path.join(P.DATA_FOLDER, 'Originals/Validation/Label3','*.tif')))

	if shuffle:
		 map(np.random.shuffle, [benign, dcis, idc])

	return benign, dcis, idc

def mask_folder():
	if P.N_CLASSES == 2:
		if 'L1' in P.DATA_FOLDER:
			path = os.path.join(P.DATA_FOLDER, 'Masks/MasksAllBinary')
		else:
			path = os.path.join(P.DATA_FOLDER, 'Masks/MasksAll2class')
	elif P.N_CLASSES == 3:
		if 'Aug17' in P.DATA_FOLDER:
			path = os.path.join(P.DATA_FOLDER, 'Masks/MasksAll')
		else:
			path = os.path.join(P.DATA_FOLDER, 'Masks/MasksAll3Class')
	return path

def per_class_filelist(Benign_file_list, DCIS_file_list, IDC_file_list, msk_fls_All, msk_src, num_each_class):
	random_Samples = []
	random_Samples = (Benign_file_list[:num_each_class[0]] +
		DCIS_file_list[:num_each_class[1]] +
		IDC_file_list[:num_each_class[2]] )

	# get rid of \m at the end of the list elements
	random_Samples = map(lambda s: s.strip(), random_Samples)

	if 'Aug17' in P.DATA_FOLDER:
		mask_extension = '_Mask.png'
	else:
		mask_extension = '_Mask.tif'

	msk_fls = []
	for ip in random_Samples:
		ip = ip.rstrip('\n')
		base_name = os.path.splitext(ntpath.basename(ip))[0]
		msk_fls.append(os.path.join(msk_fls_All, base_name + mask_extension))

	for i in range(len(random_Samples)):
		msk_src[random_Samples[i]] = msk_fls[i]

	return random_Samples, msk_src

def label_name(label):
	try:
		return label_names[label]
	except:
		return label_names[np.argmax(label)]
