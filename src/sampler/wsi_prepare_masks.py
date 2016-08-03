import joblib
import wsi_mask
import dataset
import wsi_mask
import wsi_sampler

files = dataset.train_filenames()
len(files[0]), len(files[1]), len(files[2]) #Sanity check

# Nones are for no subset, we want to prepare masks for all data
files, mask_sources = dataset.per_class_filelist(files[0],files[1],files[2],dataset.mask_folder(),{},(None,None,None))