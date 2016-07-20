import os
import glob
import shutil

folders = sorted(os.listdir('../models'))

print "Cleaning models without epochs {} folders".format(len(folders))

for model_name in folders:
    folder = '../models/'+model_name
    if len(glob.glob(folder+'/*.npz')) == 0:
        print "None in ", model_name
        shutil.rmtree(folder)

folders = sorted(os.listdir('../models'))
print "Folders remaining: {}".format(len(folders))
