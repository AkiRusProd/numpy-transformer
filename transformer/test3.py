from multiprocessing import Process, Pool
import time
import ray
import numpy as np

ray.init()

# @ray.remote
def func1(x):
	print ('func1: starting')
	for i in range(100):
		# x[i] *= 2.5353
		x @ np.random.rand(len(x), len(x))
	print ('func1: finishing')
	# return x
	
# @ray.remote
def func2(x):
	print ('func2: starting')
	for i in range(100):
		# x[i] *= x[i]
		x @ np.random.rand(len(x), len(x))
	print ('func2: finishing')
	# return x
	# print ('func2: finishing')

if __name__ == '__main__':
	x = np.random.normal(0, 1, (2000, 2000))
	start_time = time.time()
	p1 = Process(target=func1, args=(x,))
	p2 = Process(target=func2, args=(x,))
	p1.start()
	p2.start()
	p1.join()
	p2.join()
	# pool = Pool()
	# result1 = pool.apply_async(func1, [x])    # evaluate "solve1(A)" asynchronously
	# result2 = pool.apply_async(func2, [x])    # evaluate "solve2(B)" asynchronously
	# answer1 = result1.get(timeout=10)
	# answer2 = result2.get(timeout=10)
	print ('time:', time.time() - start_time)

# x = np.random.normal(0, 1, (2000, 2000))
# start_time = time.time()
# # func1(x)
# # func2(x)
# ret_id1 = func1.remote(x)
# ret_id2 = func2.remote(x)
# ret1, ret2 = ray.get([ret_id1, ret_id2])
# print ('time:', time.time() - start_time)


x = np.random.normal(0, 1, (2000, 2000))
start_time = time.time()
func1(x)
func2(x)
print ('time:', time.time() - start_time)
