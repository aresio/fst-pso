import math
import logging
import random
from numpy.random import lognormal
from numpy import array, zeros, linalg
import sys
import copy
#import numpy
import subprocess

class Particle(object):

	def __init__(self):
		self.X = []
		self.V = []
		self.B = []
		self.MarkedForRestart = False
		self.CalculatedFitness = sys.float_info.max
		self.FitnessDevStandard = sys.float_info.max
		self.CalculatedBestFitness = sys.float_info.max
		self.SinceLastLocalUpdate = 0
		
		self.DerivativeFitness = 0
		self.MagnitudeMovement = 0
		self.DistanceFromBest = sys.float_info.max
		self.CognitiveFactor = 2.
		self.SocialFactor = 2.
		self.Inertia = 0.5

		# support for PPSO
		self.MaxSpeedMultiplier = .25
		self.MinSpeedMultiplier = 0
		
	def __repr__(self):
		return "<Particle %s>" % str(self.X)

	def __str__(self):
		return "\t".join(map(str, self.X))


class PSO_new(object):

	def __repr__(self):
		return str("<PSO instance "+self.ID+">")

	def __init__(self, ID=""):
		self.G = None
		self.W = None			# worst

		self.ID = ID

		self.GIndex = 0
		self.WIndex = 0

		self.Solutions = []
		self.InertiaStart = 0.9
		self.InertiaEnd 	 = 0.4
		self.Inertia 	 = 0.9
		self.CognitiveFactor = 1.9
		self.SocialFactor    = 1.9
		self.MaxVelocity  = 1.0
		self.UseRestart = False
		self.UseLog = False
		self.ProximityThreshold = 1E-1
		self.MaxIterations = sys.maxsize
		self.MaxNoUpdateIterations = sys.maxsize
		self.GoodFitness = -sys.float_info.max
		self.Iterations = 0
		self.SinceLastGlobalUpdate = 0
		self.FITNESS = None
		self.Boundaries = []
		self.save_path = ""
		self.NumberOfParticles = 0
		self.Color = "black"
		self.Nickname = "Standard"
		self.StopOnGoodFitness = False
		self.EstimatedWorstFitness = sys.float_info.max
		self.Dimensions = 0

		self.SIMULATOR_PATH = None
		self.ARGUMENTS = None
		self.COMMUNICATION_FILE = None

		# change this for parallel fitness calculation
		self.ParallelFitness = False


	def getBestIndex(self):
		index = 0
		best_fitness = self.Solutions[0].CalculatedFitness
		for n, s in enumerate(self.Solutions):
			if s.CalculatedFitness < best_fitness:
				best_fitness = s.CalculatedFitness
				index = n
		return index

	def getWorstIndex(self):
		index = 0
		worst_fitness = self.Solutions[0].CalculatedFitness
		for n, s in enumerate(self.Solutions):
			if s.CalculatedFitness > worst_fitness:
				worst_fitness = s.CalculatedFitness
				index = n
		return index

	def UpdateInertia(self):
		self.Inertia = self.InertiaStart - ( self.InertiaStart-self.InertiaEnd ) / self.MaxIterations * self.Iterations

	def Iterate(self, verbose=False):
		self.UpdatePositions()
		self.UpdateVelocities()
		self.UpdateCalculatedFitness()
		self.UpdateLocalBest(verbose)
		self.Iterations = self.Iterations + 1
		self.SinceLastGlobalUpdate = self.SinceLastGlobalUpdate + 1 

	def Solve(self, funz, verbose=False, callback=None, dump_best_fitness=None, dump_best_solution=None):

		logging.info('Launching optimization.')

		self.Iterations = 0
		if verbose:
			print("Process started")
		while( not self.TerminationCriterion(verbose=verbose) ):
			if funz!=None:	funz(self)
			self.Iterate(verbose)
			if verbose:
				print("Completed iteration %d" % (self.Iterations))

			# new: if a callback is specified, call it at regular intervals
			if callback!=None: 
				interval = callback['interval']
				function = callback['function']
				if (self.Iterations-1)%interval==0:
					function(self)
					if verbose: print (" * Callback invoked")

			# write the current best fitness 
			if dump_best_fitness!=None:
				if self.Iterations==1: 
					with open(dump_best_fitness, "w") as fo: pass # touch
				with open(dump_best_fitness, "a") as fo:
					fo.write(str(self.G.CalculatedFitness)+"\n")
	
			# write the current best solution
			if dump_best_solution!=None:
				if self.Iterations==1:
					with open(dump_best_solution, "w") as fo: pass # touch
				with open(dump_best_solution, "a") as fo:
					fo.write("\t".join(map(str, self.G.X))+"\n")

		if verbose:
			print("Process terminated, best position:", self.G.X, "with fitness",self.G.CalculatedFitness)

		logging.info('Best solution: '+str(self.G))
		logging.info('Fitness of best solution: '+str(self.G.CalculatedFitness))

		return self.G, self.G.CalculatedFitness
		
	def TerminationCriterion(self, verbose=False):

		if verbose:
			print("Iteration:", self.Iterations)
			print(", since last global update:", self.SinceLastGlobalUpdate )

		if self.StopOnGoodFitness == True:
			if self.G.CalculatedFitness < self.GoodFitness:
				if verbose:
					print("Good fitness reached!", self.G.CalculatedFitness)
				return True

		if self.SinceLastGlobalUpdate > self.MaxNoUpdateIterations:
			if verbose:
				print("Too many iterations without new global best")
			return True
		
		if self.Iterations > self.MaxIterations:
			if verbose:
				print("Maximum iterations reached")
			return True
		else:
			return False

	def Generate(self, lista, use_log=True, verbose=False): 

		ret = []
		if use_log:
			for i in range(len(lista)):
				minimo = math.log(self.Boundaries[i][0])
				massimo = math.log(self.Boundaries[i][1])
				ret.append(math.exp(minimo+(massimo-minimo)*random.random()))
		else:
			for i in range(len(lista)):
				ret.append( self.Boundaries[i][0] + (self.Boundaries[i][1]-self.Boundaries[i][0]) * random.random() )

		if verbose:
			print("Particle generated")
			print( ret)
		return ret

	def set_number_of_particles(self, n):
		self.NumberOfParticles = n
		print("Number of particles set to", n)

	def auto_create_particles(self, dim, use_log):
		if self.NumberOfParticles == 0:
			print("ERROR: it is impossible to autocreate 0 particles")
			return
		self.CreateParticles(self.NumberOfParticles, dim, use_log)
		print(self.NumberOfParticles, "particles autocreated")



	def NewGenerate(self, lista, creation_method):

		ret = []

		if creation_method['name'] == "uniform":
			for i in range(len(lista)):
				ret.append(self.Boundaries[i][0] + (self.Boundaries[i][1]-self.Boundaries[i][0]) * random.random())

		elif creation_method['name'] == "logarithmic":

			for i in range(len(lista)):
				if self.Boundaries[i][0]<0:
					minimo = -5
					massimo = math.log(self.Boundaries[i][1])
					res = math.exp(minimo+(massimo-minimo)*random.random())
					if random.random() > .5:
						res *= -1
					ret.append(res)
				else:					
					minimo = math.log(self.Boundaries[i][0])
					massimo = math.log(self.Boundaries[i][1])
					ret.append(math.exp(minimo+(massimo-minimo)*random.random()))

		elif creation_method['name'] == "normal":
			for i in range(len(lista)):
				while(True):
					if self.Boundaries[i][1]==self.Boundaries[i][0]: 
						ret.append( self.Boundaries[i][1] )
						break
					cand_position = random.gauss( (self.Boundaries[i][1]+self.Boundaries[i][0])/2,  creation_method['sigma'] )					
					if (cand_position>=self.Boundaries[i][0] and cand_position<=self.Boundaries[i][1]):
						ret.append(cand_position)
						break

		elif creation_method['name'] == "lognormal":
			for i in range(len(lista)):
				minord = math.log(self.Boundaries[i][0])
				maxord = math.log(self.Boundaries[i][1])
				media = (maxord+minord)/2.
				while(True):
					if self.Boundaries[i][1]==self.Boundaries[i][0]: 
						ret.append( self.Boundaries[i][1] )
						break
					v = lognormal(media, creation_method['sigma'])
					if v>=self.Boundaries[i][0] and v<=self.Boundaries[i][1]:
						break
				ret.append(v)

		else:
			print("Unknown particles initialization mode")

		return ret



	def NewCreateParticles(self, n, dim, creation_method={ 'name':"uniform"}, initial_guess_list=None):

		del self.Solutions [:]

		for i in range(n):

			p = Particle()		
		
			p.X = self.NewGenerate( [0]*dim, creation_method = creation_method )
			p.B = copy.deepcopy(p.X)
			p.V = list(zeros(dim))

			self.Solutions.append(p)
			
			if len(self.Solutions)==1:
				self.G = copy.deepcopy(p)
				self.G.CalculatedFitness = sys.float_info.max
				self.W = copy.deepcopy(p)
				self.W.CalculatedFitness = sys.float_info.min


		if initial_guess_list is not None:
			if len(initial_guess_list)>n:
				print ("ERROR: the number of provided initial guesses (%d) is greater than the swarm size (%d), aborting." % (len(initial_guess_list), n))
				exit(17)
			for i, ig in enumerate(initial_guess_list):
				if len(ig) != dim:
					print ("ERROR: each initial guess must have length equal to %d, aborting." % dim)
					exit(18)
				else:
					self.Solutions[i].X=copy.deepcopy(ig)
					self.Solutions[i].B=copy.deepcopy(ig)

			print (" * %d particles created, initial guesses added to the swarm." % n)
		else:
			print (" * %d particles created." % n)

		print (" * FST-PSO will now assess the local and global best particles.")

		self.NumberOfParticles = n

		if not self.ParallelFitness:
			self.UpdateCalculatedFitness()		# experimental

		vectorFirstFitnesses = [ x.CalculatedFitness for x in self.Solutions ]
		self.EstimatedWorstFitness = max(vectorFirstFitnesses)

		self.UpdateLocalBest()
		self.UpdatePositions()	

		self.Dimensions = dim

		logging.info('%d particles created.' % (self.numberofparticles))




	def CreateParticles(self, n, dim, use_log=False):

		self.UseLog = use_log

		if self.FITNESS == None:
			print ("ERROR: particles must be created AFTER the definition of the fitness function")
			exit()

		del self.Solutions [:]

		# for all particles
		for i in range(n):

			p = Particle()
	
			p.X = self.Generate( [0]*dim, use_log = use_log )
			p.B = copy.deepcopy(p.X)
			p.V = list(zeros(dim))

			self.Solutions.append(p)
			
			if len(self.Solutions)==1:
				self.G = copy.deepcopy(p)
				self.G.CalculatedFitness = sys.float_info.max
				self.W = copy.deepcopy(p)
				self.W.CalculatedFitness = sys.float_info.min

		print( " *", n, "particles created, verifying local and global best")

		self.NumberOfParticles = n

		self.UpdateCalculatedFitness()		# experimental

		vectorFirstFitnesses = [ x.CalculatedFitness for x in self.Solutions ]
		self.EstimatedWorstFitness = max(vectorFirstFitnesses)

		self.UpdateLocalBest()

		self.UpdatePositions()

		self.Dimensions = dim

	def UpdateCalculatedFitness(self):
		for s in self.Solutions:
			prev = s.CalculatedFitness
			ret = self.FITNESS(s.X)
			if s.MagnitudeMovement!=0:
				s.DerivativeFitness = (ret-prev)/s.MagnitudeMovement
			if isinstance(ret, list):
				s.CalculatedFitness = ret[0]
				s.Differential = ret[1]
			else:
				s.CalculatedFitness = ret
		
	def UpdateLocalBest(self, verbose=False, semiverbose=True):		

		if verbose:
			print ("Starting verification of local best")
		for i in range(len(self.Solutions)):			
			if verbose:
				print (" Solution", i, ":", self.Solutions[i])
			if self.Solutions[i].CalculatedFitness < self.Solutions[i].CalculatedBestFitness:
				self.Solutions[i].SinceLastLocalUpdate = 0
				if verbose:
					print ("new best for ", i, " has fitness", self.Solutions[i].CalculatedFitness )

				self.Solutions[i].B = copy.deepcopy(self.Solutions[i].X)
				self.Solutions[i].CalculatedBestFitness = self.Solutions[i].CalculatedFitness
				if self.Solutions[i].CalculatedFitness < self.G.CalculatedFitness:
					self.G = copy.deepcopy(self.Solutions[i])
					if verbose or semiverbose:
						print ("new global best", i, "has fitness", self.Solutions[i].CalculatedFitness)

					self.SinceLastGlobalUpdate = 0			
					self.GIndex = i
			else:
				if verbose:
					print (" Fitness calcolata:", self.Solutions[i].CalculatedFitness, "old best", self.Solutions[i].CalculatedBestFitness)
				self.Solutions[i].SinceLastLocalUpdate += 1

				# update global worst
				if self.Solutions[i].CalculatedFitness > self.W.CalculatedFitness:
					self.W = copy.deepcopy(self.Solutions[i])
					self.WIndex = i

		if self.Iterations>0: 
			logging.info('[Iteration %d] best individual fitness: %f' % (self.Iterations, self.G.CalculatedFitness))
			logging.info('[Iteration %d] best individual structure: %s' % (self.Iterations, str(self.G.X)))

	# update particles' positions
	def UpdatePositions(self, constrained_damping = False):

		for p in self.Solutions:	

			if self.UseRestart:
				if p.MarkedForRestart:
					p.X = self.Generate([0]*len(p.X), use_log=self.UseLog)
					p.B = copy.deepcopy(p.X)
					p.V = list(zeros(len(p.X)))
					p.MarkedForRestart = False
					continue

			prev_pos = p.X[:]

			for n in range(len(p.X)):			
				
				c1 = p.X[n]
				c2 = p.V[n]
				tv = c1+c2
				rnd1 = rnd2 = 0
				if tv > self.Boundaries[n][1]:
					if not constrained_damping:
						rnd1 = random.random()
						tv = self.Boundaries[n][1] - rnd1 * c2
					else:
						print ("WARNING: constrained damping not implemented")

				if tv < self.Boundaries[n][0]:
					if not constrained_damping:
						rnd2 = random.random()
						tv = self.Boundaries[n][0] - rnd2 * c2
					else:
						print ("WARNING: constrained damping not implemented")

				p.X[n] = tv


	def UpdateVelocities(self):
		"""
			Update the velocity of all particles, according to the PSO settings.
		"""

		for numpart, p in enumerate(self.Solutions):

			if self.UseRestart:
				if self.getBestIndex != numpart:
					distance_from_global_best = linalg.norm( array(self.G.X) - array(p.X) )
					if distance_from_global_best<self.ProximityThreshold:
						if self.Iterations>0:	
							print (" * Particle", numpart, "marked for restart")
							p.MarkedForRestart = True

			for n in range(len(p.X)):		

				fattore1 = self.Inertia * p.V[n]
				fattore2 = random.random() * self.CognitiveFactor * (p.B[n] - p.X[n])
				fattore3 = random.random() * self.SocialFactor * (self.G.X[n] - p.X[n])

				newvelocity = fattore1+fattore2+fattore3

				if newvelocity > self.MaxVelocity[n]:
					newvelocity = self.MaxVelocity[n]
				elif newvelocity < -self.MaxVelocity[n]:
					newvelocity = -self.MaxVelocity[n]

				p.V[n] = newvelocity
