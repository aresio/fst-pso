from __future__ import print_function
import math
import pso
import logging
from numpy import random, array, linalg
import os
from miniful import *


class FuzzyPSO(pso.PSO_new):

	def __init__(self, logfile=None):
		"""
		Creates a new FST-PSO instance.

		Args:
			D: number of dimensions of the problem under optimization
		"""

		super(FuzzyPSO, self).__init__()

		# defaults for membership functions
		self.DER1 = -1.0
		self.DER2 =  1.0
		self.MDP1 = 0.2 
		self.MDP2 = 0.4 
		self.MDP3 = 0.6 
		self.MaxDistance = 0
		self.dimensions = 0

		self.enabled_settings = ["cognitive", "social", "inertia", "minvelocity", "maxvelocity"]

		self._FITNESS_ARGS = None

		if logfile!=None:
			print (" * Logging to file", logfile, "enabled")
			with open(logfile, 'w'): pass
			logging.basicConfig(filename=logfile, level=logging.DEBUG)
			logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
			logging.info('FST-PSO object created.')


	def disable_fuzzyrule_cognitive(self):
		self.enabled_settings = filter(lambda x: x != "cognitive", self.enabled_settings)
		print (" * Fuzzy rules for cognitive factor DISABLED")

	def disable_fuzzyrule_social(self):
		self.enabled_settings = filter(lambda x: x != "social", self.enabled_settings)
		print (" * Fuzzy rules for social factor DISABLED")

	def disable_fuzzyrule_inertia(self):
		self.enabled_settings = filter(lambda x: x != "inertia", self.enabled_settings)
		print (" * Fuzzy rules for inertia weight DISABLED")

	def disable_fuzzyrule_maxvelocity(self):
		self.enabled_settings = filter(lambda x: x != "maxvelocity", self.enabled_settings)
		print (" * Fuzzy rules for maximum velocity cap DISABLED")

	def disable_fuzzyrule_minvelocity(self):
		self.enabled_settings = filter(lambda x: x != "minvelocity", self.enabled_settings)
		print (" * Fuzzy rules for minimum velocity cap DISABLED")


	def solve_with_fstpso(self, max_iter=100, creation_method={'name':"uniform"}, 
		callback=None, verbose=False, arguments=None, dump_best_fitness=None, dump_best_solution=None):
		"""
			Launches the optimization using FST-PSO. Internally, this method checks
			that we correctly set the pointer to the fitness function and the
			boundaries of the search space.

			Args:
				max_iter: the maximum number of iterations of FST-PSO
				creation_method: specifies the type of particles initialization
				arguments: these arguments will be passed to the fitness evaluation function. Please note that
				           the fitness function must accept two arguments (the particle and the arguments)
				dump_best_fitness: at the end of each iteration fst-pso will save to this file the best fitness value
				dump_best_solution: at the end of each iteration fst-pso will save to this file the structure of the 
				                    best solution found so far
				callback: this argument accepts a dictionary with two items: 'function' and 'interval'. Every 
				          'interval' iterations, the 'function' is called. This functionality can be exploited 
				          to implement special behavior (e.g., restart)
				verbose: enable verbose mode

			Returns:
			    This method returns a couple (optimal solution, fitness of the optimal solution)
		"""

		if self.FITNESS is None:
			print ("ERROR: cannot solve a problem without a fitness function; use set_fitness()")
			exit(-3)

		if self.Boundaries == []:
			print ("ERROR: FST-PSO cannot solve unbounded problems; use set_search_space()")
			exit(-4)

		self.MaxIterations = max_iter
		print (" * Max iterations set to", self.MaxIterations)
		print (" * Settings:", self.enabled_settings)
		print (" * Launching optimization")

		self.NewCreateParticles(self.numberofparticles, self.dimensions, creation_method=creation_method)

		result = self.Solve(None, verbose=verbose, callback=callback, dump_best_solution=dump_best_solution, dump_best_fitness=dump_best_fitness)
		
		return result


	def set_search_space(self, limits):
		D = len(limits)
		self.dimensions = D

		self.MaxDistance = max_distance=calculate_max_distance(limits)
		print (" * Max distance: %f" % self.MaxDistance)

		self.Boundaries = limits		
		print (" * Search space boundaries set to:", limits)

		self.MaxVelocity = [  math.fabs(B[1]-B[0]) for B in limits ]
		print (" * Max velocities set to:", self.MaxVelocity)

		self.numberofparticles = int(10 + 2*math.sqrt(D))
		print (" * Number of particles automatically set to", self.numberofparticles)

		logging.info('Search space set (%d dimensions).' % (D))


	def set_swarm_size(self, N):
		"""
			This (optional) method overrides FST-PSO's heuristic and forces the number of particles in the swarm.

		"""
		try:
			N=int(N)
		except:
			print ("ERROR: please specify the swarm size as an integer number")
			exit(-6)

		if N<=1:
			print ("ERROR: FST-PSO cannot work with less than 1 particles, aborting")
			exit(-5)
		else:
			self.numberofparticles = N
			print (" * Swarm size now set to %d particles" % (self.numberofparticles))

		logging.info('Swarm size set to %d particles.' % (self.numberofparticles))


	def call_fitness(self, data, arguments=None):
		if arguments==None:	return self.FITNESS(data)
		else:				return self.FITNESS(data, arguments)


	def set_fitness(self, fitness, arguments=None, skip_test=False):			
		if skip_test:
			self.FITNESS = fitness
			self._FITNESS_ARGS = arguments
			self.ParallelFitness = False
			return 

		try:
			print (" * Testing fitness evaluation")
			self.FITNESS = fitness
			self._FITNESS_ARGS = arguments
			self.call_fitness([1e-10]*self.dimensions, self._FITNESS_ARGS)
			self.ParallelFitness = False
			print (" * Test successful")
		except:
			print ("ERROR: the specified function does not seem to implement a correct fitness function")
			exit(-2)


	def set_parallel_fitness(self, fitness, arguments=None, skip_test=False):
		if skip_test:
			self.FITNESS = fitness
			self._FITNESS_ARGS = arguments
			self.ParallelFitness = True
			return 

		np = pso.Particle()
		np.X = [0]*self.dimensions
		fitness([np.X]*self.numberofparticles)
		self.FITNESS = fitness
		self._FITNESS_ARGS = arguments
		self.ParallelFitness = True


	def phi(self, f_w, f_o, f_n, phi, phi_max):
		""" 
			Calculates the Fitness Incremental Factor (phi).
		"""

		if phi == 0:
			return 0
		denom = (min(f_w, f_n) - min(f_w, f_o))/f_w			# 0..1
		numer = phi/phi_max									# 0..1		
		return denom*numer


	def CreateFuzzyReasoner(self, max_delta):

		FR = FuzzyReasoner()

		p1 = max_delta*self.MDP1
		p2 = max_delta*self.MDP2
		p3 = max_delta*self.MDP3


		LOW_INERTIA = 0.3
		MEDIUM_INERTIA = 0.5
		HIGH_INERTIA = 1.0

		LOW_SOCIAL = 1.0
		MEDIUM_SOCIAL = 2.0
		HIGH_SOCIAL = 3.0

		LOW_COGNITIVE = 0.1
		MEDIUM_COGNITIVE = 1.5
		HIGH_COGNITIVE = 3.0

		LOW_MINSP = 0
		MEDIUM_MINSP = 0.001
		HIGH_MINSP = 0.01

		LOW_MAXSP = 0.1
		MEDIUM_MAXSP = 0.15
		HIGH_MAXSP = 0.2


		myFS1 = FuzzySet(points=[[0, 0],	[1., 1.], 	[1., 0]], 	term="WORSE")
		myFS2 = FuzzySet(points=[[-1., 0],	[0, 1.],	[1., 0]], 	term="SAME")
		myFS3 = FuzzySet(points=[[-1., 0],	[-1., 1.],	[0, 0]], 	term="BETTER")
		PHI_MF = MembershipFunction( [myFS1, myFS2, myFS3], concept="PHI" )

		myFS4 = FuzzySet(points=[[0, 0],	[0, 1.], 	[p1, 1.], [p2, 0]], 	term="SAME")
		myFS5 = FuzzySet(points=[[p1, 0],	[p2, 1.],	[p3, 0]], 	term="NEAR")
		myFS6 = FuzzySet(points=[[p2, 0],	[p3, 1.],	[max_delta, 1.]], 	term="FAR")
		DELTA_MF = MembershipFunction( [myFS4, myFS5, myFS6], concept="DELTA" )


		myR1 = FuzzyRule( IF(PHI_MF, "WORSE"), 	THEN("INERTIA", LOW_INERTIA), comment="Rule inertia worse phi" )
		myR2 = FuzzyRule( IF(PHI_MF, "WORSE"),	THEN("SOCIAL", HIGH_SOCIAL), comment="Rule social worse phi" )
		myR3 = FuzzyRule( IF(PHI_MF, "WORSE"),	THEN("COGNITIVE", MEDIUM_COGNITIVE), comment="Rule cognitive worse phi" )
		myR4 = FuzzyRule( IF(PHI_MF, "WORSE"), 	THEN("MINSP", HIGH_MINSP), comment="Rule min speed worse phi" )
		myR5 = FuzzyRule( IF(PHI_MF, "WORSE"), 	THEN("MAXSP", HIGH_MAXSP), comment="Rule max speed worse phi" )

		myR6 = FuzzyRule( IF(PHI_MF, "SAME"), 	THEN("INERTIA", MEDIUM_INERTIA), comment="Rule inertia same phi" )
		myR7 = FuzzyRule( IF(PHI_MF, "SAME"),	THEN("SOCIAL", MEDIUM_SOCIAL), comment="Rule social same phi" )
		myR8 = FuzzyRule( IF(PHI_MF, "SAME"),	THEN("COGNITIVE", MEDIUM_COGNITIVE), comment="Rule cognitive same phi" )
		myR9 = FuzzyRule( IF(PHI_MF, "SAME"), 	THEN("MINSP", LOW_MINSP), comment="Rule min speed same phi" )
		myR10 = FuzzyRule( IF(PHI_MF, "SAME"), 	THEN("MAXSP", MEDIUM_MAXSP), comment="Rule max speed same phi" )

		myR11 = FuzzyRule( IF(PHI_MF, "BETTER"), 	THEN("INERTIA", HIGH_INERTIA), comment="Rule inertia better phi" )
		myR12 = FuzzyRule( IF(PHI_MF, "BETTER"),	THEN("SOCIAL", LOW_SOCIAL), comment="Rule social better phi" )
		myR13 = FuzzyRule( IF(PHI_MF, "BETTER"),	THEN("COGNITIVE", HIGH_COGNITIVE), comment="Rule cognitive better phi" )
		myR14 = FuzzyRule( IF(PHI_MF, "BETTER"), 	THEN("MINSP", LOW_MINSP), comment="Rule min speed better phi" )
		myR15 = FuzzyRule( IF(PHI_MF, "BETTER"), 	THEN("MAXSP", MEDIUM_MAXSP), comment="Rule max speed better phi" )


		myR16 = FuzzyRule( IF(DELTA_MF, "SAME"), 	THEN("INERTIA", LOW_INERTIA), comment="Rule inertia same delta" )
		myR17 = FuzzyRule( IF(DELTA_MF, "SAME"),	THEN("SOCIAL", MEDIUM_SOCIAL), comment="Rule social same delta" )
		myR18 = FuzzyRule( IF(DELTA_MF, "SAME"),	THEN("COGNITIVE", MEDIUM_COGNITIVE), comment="Rule cognitive same delta" )
		myR19 = FuzzyRule( IF(DELTA_MF, "SAME"), 	THEN("MINSP", MEDIUM_MINSP), comment="Rule min speed same delta" )
		myR20 = FuzzyRule( IF(DELTA_MF, "SAME"), 	THEN("MAXSP", LOW_MAXSP), comment="Rule max speed same delta" )

		myR21 = FuzzyRule( IF(DELTA_MF, "NEAR"), 	THEN("INERTIA", MEDIUM_INERTIA), comment="Rule inertia near delta" )
		myR22 = FuzzyRule( IF(DELTA_MF, "NEAR"),	THEN("SOCIAL", LOW_SOCIAL), comment="Rule social near delta" )
		myR23 = FuzzyRule( IF(DELTA_MF, "NEAR"),	THEN("COGNITIVE", MEDIUM_COGNITIVE), comment="Rule cognitive near delta" )
		myR24 = FuzzyRule( IF(DELTA_MF, "NEAR"), 	THEN("MINSP", MEDIUM_MINSP), comment="Rule min speed near delta" )
		myR25 = FuzzyRule( IF(DELTA_MF, "NEAR"), 	THEN("MAXSP", MEDIUM_MAXSP), comment="Rule max speed near delta" )

		myR26 = FuzzyRule( IF(DELTA_MF, "FAR"), 	THEN("INERTIA", LOW_INERTIA), comment="Rule inertia far delta" )
		myR27 = FuzzyRule( IF(DELTA_MF, "FAR"),		THEN("SOCIAL", MEDIUM_SOCIAL), comment="Rule social far delta" )
		myR28 = FuzzyRule( IF(DELTA_MF, "FAR"),		THEN("COGNITIVE", MEDIUM_COGNITIVE), comment="Rule cognitive far delta" )
		myR29 = FuzzyRule( IF(DELTA_MF, "FAR"), 	THEN("MINSP", MEDIUM_MINSP), comment="Rule min speed far delta" )
		myR30 = FuzzyRule( IF(DELTA_MF, "FAR"), 	THEN("MAXSP", LOW_MAXSP), comment="Rule max speed far delta" )


		FR.add_rules([myR1, myR2, myR3, myR4, myR5, myR6, myR7, myR8, myR9, myR10,
						myR11, myR12, myR13, myR14, myR15, myR16, myR17, myR18, myR19, myR20,
						myR21, myR22, myR23, myR24, myR25, myR26, myR27, myR28, myR29, myR30 ])

		return FR


	def UpdateCalculatedFitness(self):
		"""
			Calculate the fitness values for each particle according to user's fitness function,
			and then update the settings of each particle.
		"""

		if self.ParallelFitness:
			#print " * Distributing calculations..."
			ripop = map(lambda x: x.X, self.Solutions)
			all_fitness = self.FITNESS(ripop, self._FITNESS_ARGS)
		else:
			all_fitness = []
			for s in self.Solutions:
				# all_fitness.append( self.FITNESS(s.X, self._FITNESS_ARGS ) )
				all_fitness.append( self.call_fitness(s.X, self._FITNESS_ARGS ) )

		# for each i-th individual "s"...
		for i,s in enumerate(self.Solutions):

			prev = s.CalculatedFitness
			ret = all_fitness[i]
			if s.MagnitudeMovement != 0:
				s.DerivativeFitness = (ret-prev)/s.MagnitudeMovement

			s.NewDerivativeFitness = self.phi(self.EstimatedWorstFitness, prev, ret, s.MagnitudeMovement, self.MaxDistance)

			if isinstance(ret, list):
				s.CalculatedFitness = ret[0]
				s.Differential = ret[1]
			else:
				s.CalculatedFitness = ret

			"""
			my_input = { "Derivative": s.NewDerivativeFitness, "Distance": s.DistanceFromBest }
			my_output = { "Inertia": 0.0, "Social": 0.0, "Cognitive": 0.0, "Maxspeed": 0.0, "Sigma": 0.0 }


			try:
				my_output = sugeno_inference(my_input, self.fuzzySystem, verbose=False)
			except:
				print my_input
				exit(-100)

			if "cognitive" in self.enabled_settings: 	s.CognitiveFactor = my_output["Cognitive"]
			if "social" in self.enabled_settings: 		s.SocialFactor = my_output["Social"]
			if "inertia" in self.enabled_settings: 		s.Inertia = my_output["Inertia"]
			if "maxvelocity" in self.enabled_settings: 	s.MaxSpeedMultiplier = my_output["Maxspeed"]		# because velocities are vectorial
			if "minvelocity" in self.enabled_settings: 	s.MinSpeedMultiplier = my_output["Sigma"]			# because velocities are vectorial
			"""

			####### TEST #######
			
			FR = self.CreateFuzzyReasoner(self.MaxDistance)
			FR.set_variable("PHI", s.NewDerivativeFitness)
			FR.set_variable("DELTA", s.DistanceFromBest)
			res = FR.evaluate_rules()

			
			if "cognitive" in self.enabled_settings: 	s.CognitiveFactor 	= res["COGNITIVE"]
			if "social" in self.enabled_settings: 		s.SocialFactor 		= res["SOCIAL"]
			if "inertia" in self.enabled_settings: 		s.Inertia 			= res["INERTIA"]
			if "maxvelocity" in self.enabled_settings: 	s.MaxSpeedMultiplier = res["MAXSP"]
			if "minvelocity" in self.enabled_settings: 	s.MinSpeedMultiplier = res["MINSP"]



	def UpdatePositions(self):
		"""
			Update particles' positions and update the internal information.
		"""

		for p in self.Solutions:	

			prev_pos = p.X[:]

			for n in range(len(p.X)):							
				c1 = p.X[n]
				c2 = p.V[n]
				tv = c1+c2
				rnd1 = rnd2 = 0
				if tv > self.Boundaries[n][1]:
					rnd1 = random.random()
					tv = self.Boundaries[n][1] - rnd1 * c2
				if tv < self.Boundaries[n][0]:
					rnd2 = random.random()
					tv = self.Boundaries[n][0] - rnd2 * c2

				p.X[n] = tv

			p.MagnitudeMovement = linalg.norm(array(p.X)-array(prev_pos), ord=2)
			p.DistanceFromBest = linalg.norm(array(p.X)-array(self.G.X), ord=2)

		#logging.info('Particles positions updated.')

	def UpdateVelocities(self):
		"""
			Update the velocity of all particles, according to their own settings.
		"""

		for p in self.Solutions:

			for n in range(len(p.X)):

				fattore1 = p.Inertia * p.V[n]
				fattore2 = random.random() * p.CognitiveFactor * (p.B[n] - p.X[n])
				fattore3 = random.random() * p.SocialFactor * (self.G.X[n] - p.X[n])

				newvelocity = fattore1+fattore2+fattore3

				# check max vel
				if newvelocity > self.MaxVelocity[n] * p.MaxSpeedMultiplier:
					newvelocity = self.MaxVelocity[n] * p.MaxSpeedMultiplier 
				elif newvelocity < -self.MaxVelocity[n] * p.MaxSpeedMultiplier:
					newvelocity = -self.MaxVelocity[n] * p.MaxSpeedMultiplier 

				# check min vel
				if abs(newvelocity) < self.MaxVelocity[n] * p.MinSpeedMultiplier:
					newvelocity = math.copysign(self.MaxVelocity[n] * p.MinSpeedMultiplier, newvelocity)

				# finally set velocity
				p.V[n] = newvelocity

		#logging.info('Particles velocities updated.')

def calculate_max_distance(interval):
	accum = 0
	for i in interval:
		accum += (i[1]-i[0])**2
	return math.sqrt(accum)


if __name__ == '__main__':
	
	print ("ERROR: please create a new FuzzyPSO object, specify a fitness function and the search space")
