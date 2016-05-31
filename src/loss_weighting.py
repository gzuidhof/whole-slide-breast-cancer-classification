from __future__ import division
import numpy as np

def weight_by_class_balance(truth, classes=None, eps=1e-8):
    
    if classes is None:
        classes = np.unique(truth)
    
    weight_map = np.zeros_like(truth, dtype=np.float32)
    
    total_amount = np.product(truth.shape)
    
    for c in classes:
        class_mask = np.where(truth==c,1,0)
        class_weight = 1/(np.sum(class_mask)/total_amount)
        
        weight_map += (class_mask*class_weight)/total_amount
        print c
        
    return weight_map

if __name__ == "__main__":
    x = np.array([[1,0,0], [-1, 1, 0]])
    print weight_by_class_balance(x, [0,1])