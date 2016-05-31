import numpy as np

def calc_errors(truth, prediction, class_number=1):
    tp = np.sum(np.equal(truth,class_number)*np.equal(prediction,class_number))
    tn = np.sum(np.not_equal(truth,class_number)*np.not_equal(prediction,class_number))
    
    fp = np.sum(np.not_equal(truth,class_number)*np.equal(prediction,class_number))
    fn = np.sum(np.equal(truth,class_number)*np.not_equal(prediction,class_number))

    return tp, tn, fp, fn
