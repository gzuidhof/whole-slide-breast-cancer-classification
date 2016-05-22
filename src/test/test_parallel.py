import sys
sys.path.append('../')
import time

from parallel import ParallelBatchIterator


#TEST_DATA = 'abcdefghijlkmnopqrstuvwxyz'.split()


def test_in_order():
    def gen(job):
        time.sleep((10-job)*0.001)
        return job
        
    pbg = ParallelBatchIterator(gen, X=range(10), ordered=False, batch_size=1)
    
    batches = []
    for batch in pbg:
        print batch
        batches.append(batch)
        
    assert(batches == range(10))
    
    

