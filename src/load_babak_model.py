import joblib
import numpy as np
import lasagne

folder = '/media/resfilsp10/pathology/Userdata/babak/Deeplearning/BabakGit/DeepLearningLasange/models/'
filename = folder + 'WRN_L1_Color_Flip_Oberon_NoReg_2016-08-09_16.34/WRN_L1_Color_Flip_Oberon_NoReg_2016-08-09_16.34_best_epoch.pkl'
goal_filename = '../models/wide_resnet_babak.npz'

network = joblib.load(filename)



np.savez_compressed(goal_filename, *lasagne.layers.get_all_param_values(network))