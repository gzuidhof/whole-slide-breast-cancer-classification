import os
import glob
import shutil

folders = sorted([name for name in os.listdir('../models') if os.path.isdir(os.path.join('../models', name))])

print "Cleaning models without epochs {} folders".format(len(folders))

for model_name in folders:
    folder = '../models/'+model_name
    if len(glob.glob(folder+'/*.npz')) == 0:
        print "None in ", model_name, ".. removing"
        shutil.rmtree(folder)

folders = sorted(os.listdir('../models'))
print "Files remaining: {}".format(len(folders))
