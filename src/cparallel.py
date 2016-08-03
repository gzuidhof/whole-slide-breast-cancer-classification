from __future__ import division
import math
from multiprocessing import Process, Queue, JoinableQueue, Value, Event
from threading import Thread
from functools import partial
import math

class ContinuousParallelBatchIterator(object):
	"""
	Uses a producer-consumer model to prepare batches on the CPU in different processes or threads (while you are training on the GPU).
	Continuous version, continues between and after batches.
	


	Constructor arguments:
		batch_generator: function which can be called to yield a new batch.
		
		ordered: boolean (default=False), whether the order of the batches matters
		batch_size: integer (default=1), amount of points in one batch
		
		multiprocess: boolean (default=True), multiprocess instead of multithrea
		n_producers: integer (default=4), amount of producers (threads or processes)
		max_queue_size: integer (default=4*n_producers)
	"""

	def __init__(self, batch_generator, batch_size=1, ordered=False, multiprocess=True, n_producers=4, max_queue_size=None):
		self.generator = batch_generator
		self.ordered = ordered
		self.multiprocess = multiprocess
		self.n_producers = n_producers
		self.batch_size = batch_size

		if max_queue_size is None:
			self.max_queue_size = n_producers*4
		else:
			self.max_queue_size = max_queue_size

		self.job_queue = Queue()
		self.last_retrieved_job = 0
		self.last_added_job = 0
		self.started = False

	def start(self):
		self.queue = JoinableQueue(maxsize=self.max_queue_size)
	
		# Flag used for keeping values in completed queue in order
		self.last_completed_job = Value('i', -1)
		self.exit = Event()

		# Start worker processes or threads
		for i in xrange(self.n_producers):
			name = "ParallelBatchIterator worker {0}".format(i)
		
			if self.multiprocess:
				p = Process(target=_produce_helper, args=(i,self.generator, self.job_queue, self.queue, self.last_completed_job, self.ordered, self.exit), name=name)
			else:
				p = Thread(target=_produce_helper, args=(i,self.generator, self.job_queue, self.queue, self.last_completed_job, self.ordered, self.exit), name=name)

			# Make the process daemon, so the main process can die without these finishing
			p.daemon = True
			p.start()

		self.started = True

	def append(self, todo):
		for job in chunks(todo, self.batch_size):
			self.job_queue.put((self.last_added_job, job))
			self.last_added_job += 1

	def __call__(self, n_batches, X=None):
		if X is not None:
			self.append(X)

		if not self.started:
			self.start()

		n_upcoming_batches = self.last_added_job - self.last_retrieved_job

		if n_upcoming_batches < n_batches:
			print "Not enough X appended to retrieve this many batches"
			print "Returning the maximum amount instead ({})".format(n_upcoming_batches)
			n_batches = n_upcoming_batches


		return GeneratorLen(self.__gen_batch(n_batches), n_batches)

	def __gen_batch(self, n_batches):
		# Run as consumer (read items from queue, in current thread)
		for x in xrange(n_batches):
			item = self.queue.get()
			self.last_retrieved_job += 1
			if item is not None:
				yield item # Yield the item to the consumer (user)
			self.queue.task_done()

	def stop(self):
		self.exit.set()
		self.queue.close()
		self.job_queue.close()

class GeneratorLen(object):
	def __init__(self, gen, length):
		self.gen = gen
		self.length = length

	def __len__(self): 
		return self.length

	def __iter__(self):
		return self.gen

def _produce_helper(id, generator, jobs, result_queue, last_completed_job, ordered, exit):
	"""
		What one worker executes, defined as a top level function as this is required for the Windows platform.
	"""
	while not exit.is_set():
		job_index, task = jobs.get()

		try:
			result = generator(task)
		except:
			print "Producer failed, skipping"
			result = None

		# Put result onto the 'done'-queue
		while not exit.is_set():
			# My turn to add job result (to keep it in order)?
			if not ordered or last_completed_job.value == job_index-1:
				with last_completed_job.get_lock():
					result_queue.put(result)
					last_completed_job.value += 1
					break
		

def chunks(l, n):
	""" Yield successive n-sized chunks from l.
		from http://goo.gl/DZNhk
	"""
	for i in xrange(0, len(l), n):
		yield l[i:i+n]
