import sys
import time

# Class to report basic results of an evolutionary algorithm
class Reporter:

	def __init__(self, filename):
		self.allowedTime = 300
		self.numIterations = 0
		self.filename = filename + ".csv"
		self.delimiter = ','
		self.startTime = time.time()
		self.writingTime = 0
		outFile = open(self.filename, "w")
		outFile.write("# Student number: " + filename + "\n")
		outFile.write("# Iteration, Elapsed time, Mean value, Best value, Cycle\n")
		outFile.close()
		self.verbose = True

	# Append the reported mean objective value, best objective value, and the best tour
	# to the reporting file. 
	#
	# Returns the time that is left in seconds as a floating-point number.
	def report(self, meanObjective, bestObjective, bestSolution):
		if (time.time() - self.startTime < self.allowedTime + self.writingTime):
			start = time.time()
			
			outFile = open(self.filename, "a")
			outFile.write(str(self.numIterations) + self.delimiter)
			outFile.write(str(start - self.startTime - self.writingTime) + self.delimiter)
			outFile.write(str(meanObjective) + self.delimiter)
			outFile.write(str(bestObjective) + self.delimiter)
			

			for i in range(bestSolution.size-1):
				outFile.write(str(bestSolution[i]) + "-")
			outFile.write(str(bestSolution[bestSolution.size-1]))
			outFile.write('\n')
			outFile.close()

			self.numIterations += 1
			self.writingTime += time.time() - start
			
			if self.verbose and self.numIterations % 10 == 0:
				print(f"Iteration {self.numIterations}: Mean Obj = {int(meanObjective)}, Best Obj = {int(bestObjective)}", end='\r')
		return (self.allowedTime + self.writingTime) - (time.time() - self.startTime)

