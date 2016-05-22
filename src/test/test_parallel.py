import sys
sys.path.append('../')
import time

from parallel import ParallelBatchIterator

def test_in_order():
    # Job that takes longer if it's an earlier job
    def gen(job):
        time.sleep((10-job)*0.001)
        return job

    jobs = range(10)
    pbg = ParallelBatchIterator(gen, X=jobs, ordered=True, batch_size=1, multiprocess=False)

    batches = []
    for batch in pbg:
        batches.append(batch)

    # Batches are delivered in an ordered manner
    assert(batches == range(10))

def test_chunking():

    gen = lambda j: j
    jobs = range(5)

    pbg = ParallelBatchIterator(gen, X=jobs, ordered=True, batch_size=2, multiprocess=False)

    batches = []
    for batch in pbg:
        batches.append(batch)

    # Batches are sized as expected, residu is in last batch
    assert(batches == [[0,1],[2,3],[4]])



if __name__ == "__main__":
    test_in_order()
    test_chunking()
