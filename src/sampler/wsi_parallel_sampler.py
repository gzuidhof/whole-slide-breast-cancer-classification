import numpy as np
import multiresolutionimageinterface as mir
from multiprocessing import Process, Queue, JoinableQueue, Value

from threading import Thread
from functools import partial
import time

class WSIParallelSampler(object):
    """
    Sampler object, uses multiresolutionimageinterface for sampling.
    """

    def __init__(self, filename, data_level, X=None, multiprocess=True, n_producers=2, max_queue_size=None):
        """
            filename string:path to the WSI image (generally a .tif or .mrxs)
            data_level: integer data level to load patches from
        """
        self.filename = filename
        self.data_level = data_level
        self.dims = None

        assert data_level > -1
        self.multiprocess = multiprocess
        self.n_producers = n_producers

        if max_queue_size is None:
            self.max_queue_size = n_producers*4
        else:
            self.max_queue_size = max_queue_size

    def __len__(self):
        return len(self.X)
    

    def __iter__(self):
        
        # Call set_todo_positions first!
        assert self.X is not None
        queue = JoinableQueue(maxsize=1)

        job_queue = self._start_producers(queue)

        # Run as consumer (read items from queue, in current thread)
        
        cache_queue = JoinableQueue(maxsize=self.max_queue_size)
        
        def batcher(queue, cache_queue):
            for x in xrange(len(self.X)):
                job_index, item = queue.get()
                cache_queue.put((job_index, item))
                time.sleep(0.1)
                queue.task_done()
                #print 'batched', x
            
        
        
        p = Thread(target=batcher, args=(queue,cache_queue), name='Batcher')
        # Make the thread daemon, so the main process can die without these finishing
        p.daemon = True
        p.start()

        
        
        for x in xrange(len(self.X)):
            job_index, item = cache_queue.get()
            #print "GET", job_index
            if job_index != x:
                print "Wrong order!"
            yield item # Yield the item to the consumer (user)
            cache_queue.task_done()

        queue.close()
        job_queue.close()
        cache_queue.close()


    def _start_producers(self, result_queue):
        n_workers = self.n_producers
        jobs = Queue()

        # Flag used for keeping values in queue in order
        last_queued_job = Value('i', -1)
        self.order_lock = last_queued_job

        # Add jobs to queue
        for job_index, x in enumerate(self.X):
            jobs.put( (job_index,x) )
            
        # Add poison pills to queue (to signal workers to stop)
        for i in xrange(n_workers):
            jobs.put((-1,None))

        produce = partial(_produce_helper,
            filename=self.filename,
            data_level=self.data_level,
            jobs=jobs,
            result_queue=result_queue,
            last_queued_job=last_queued_job)

        # Start worker processes or threads
        for i in xrange(n_workers):
            name = "ParallelSampler worker {0}".format(i)
        
            if self.multiprocess:
                p = Process(target=produce, args=(i,), name=name)
            else:
                p = Thread(target=produce, args=(i,), name=name)

            # Make the process daemon, so the main process can die without these finishing
            p.daemon = True
            p.start()

        return jobs
    
    def get_image_dimensions(self):
        r = mir.MultiResolutionImageReader()
        img = r.open(self.filename)
        dims = img.getLevelDimensions(self.data_level)
        img.close()
        return dims
    
    # Tuples of X, Y, width, height
    # where to extract areas in WSI.
    def set_todo_positions(self, X):
        self.X = X

def _produce_helper(id, filename, data_level, jobs, result_queue, last_queued_job):
    """
        What one worker executes, defined as a top level function as this is required for the Windows platform.
    """

    # Open the image
    r = mir.MultiResolutionImageReader()
    img = r.open(filename)

    while True:
        job_index, task = jobs.get()

        # Kill the worker if there is no more work
        # (This is a poison pill)
        if job_index == -1 and task is None:
            img.close()
            break

        x, y, width, height = task
        image = img.getUCharPatch(x,y, width, height, data_level)
        
        result = (job_index, image.transpose(2,0,1)) 

        # Put result onto the 'done'-queue
        while True:
            # My turn to add job result (to keep it in order)?
            if last_queued_job.value == job_index-1:
                with last_queued_job.get_lock():
                    result_queue.put(result)
                    last_queued_job.value += 1
                    #print "placed", job_index
                    break