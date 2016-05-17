from multiprocessing import Process, Queue, JoinableQueue, Value
#from multiprocess import Process, Queue, JoinableQueue, Value
from threading import Thread
import util

from functools import partial

class ParallelBatchIterator(object):
	"""
	Uses a producer-consumer model to prepare batches on the CPU while training on the GPU.

	Constructor arguments:
		batch_generator: function which can be called to yield a new batch.
		ordered: boolean, whether the order of the batches matters

		batch_size: amount of points in one batch
		multiprocess: multiprocess instead of multithread
		n_producers: amount of producers (threads of processes)
		max_queue_size: optional, default 2*n_producers

	"""

	def __init__(self, batch_generator, X, batch_size=1, ordered=False, multiprocess=False, n_producers=1, max_queue_size=None):
		self.generator = batch_generator
		self.ordered = ordered
		self.multiprocess = multiprocess
		self.n_producers = n_producers
		self.X = X
		self.batch_size = batch_size

		if max_queue_size is None:
			self.max_queue_size = n_producers*2
		else:
			self.max_queue_size = max_queue_size

	def __call__(self):
		return self

	def __iter__(self):
		queue = JoinableQueue(maxsize=self.max_queue_size)

		n_batches, job_queue = self.start_producers(queue)

		# Run as consumer (read items from queue, in current thread)
		for x in xrange(n_batches):
			item = queue.get()
			#print len(item[0]), queue.qsize(), "GET"
			yield item
			queue.task_done()

		#queue.join() #Lock until queue is fully done
		queue.close()
		job_queue.close()

	def start_producers(self, result_queue):
		jobs = Queue()
		n_workers = self.n_producers
		batch_count = 0

		#Flag used for keeping values in queue in order
		last_queued_job = Value('i', -1)

		for job_index, batch in enumerate(util.chunks(self.X,self.batch_size)):
			batch_count += 1
			jobs.put( (job_index,batch) )

		# Define producer (putting items into queue)
		produce = partial(produce_helper,
			generator=self.generator,
			jobs=jobs,
			result_queue=result_queue,
			last_queued_job=last_queued_job,
			ordered=self.ordered)

		# Start workers
		for i in xrange(n_workers):

			if self.multiprocess:
				p = Process(target=produce, args=(i,))
			else:
				p = Thread(target=produce, args=(i,))

			p.daemon = True
			p.start()

		# Add poison pills to queue (to signal workers to stop)
		for i in xrange(n_workers):
			jobs.put((-1,None))

		return batch_count, jobs

def produce_helper(id, generator, jobs, result_queue, last_queued_job, ordered):
	while True:
		job_index, task = jobs.get()

		if task is None:
			#print id, " fully done!"
			break

		result = generator(*task)

		while(True):
			# My turn to add job result?
			if last_queued_job.value == job_index-1 or not ordered:

				with last_queued_job.get_lock():
					result_queue.put(result)
					last_queued_job.value += 1
					#print id, " worker PUT", job_index
					break
