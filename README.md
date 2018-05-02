# FST-PSO

*Fuzzy Self-Tuning PSO* (FST-PSO) is a swarm intelligence global optimization method [1]
based on Particle Swarm Optimization [2].

FST-PSO is designed for the optimization of real-valued multi-dimensional multi-modal minimization problems.

FST-PSO is settings-free version of PSO which exploits fuzzy logic to dynamically assign the functioning parameters to each particle in the swarm. Specifically, during each generation, FST-PSO is determines the optimal choice for the cognitive factor, the social factor, the inertia value, the minimum velocity, and the maximum velocity. FST-PSO also uses an heuristics to choose the swarm size, so that the user must not select any functioning setting.

In order to use FST-PSO, the programmer must implement a custom fitness function and specify the boundaries of the search space for each dimension. The programmer can optionally specify the maximum number of iterations. When the stopping criterion is met, FST-PSO returns the best fitting solution found, along with its fitness value.


## Example

FST-PSO can be used as follows:

	from fstpso import FuzzyPSO	
	
	def example_fitness( particle ):
		return sum(map(lambda x: x**2, particle))
		
	if __name__ == '__main__':
		dims = 10
		FP = FuzzyPSO()
		FP.set_search_space( [[-10, 10]]*dims )	
		FP.set_fitness(example_fitness)
		result =  FP.solve_with_fstpso()
		print "Best solution:", result[0]
		print "Whose fitness is:", result[1]


## Further information

FST-PSO has been created by M.S. Nobile, D. Besozzi, G. Pasi, G. Mauri, 
R. Colombo (University of Milan-Bicocca, Italy), and P. Cazzaniga (University
of Bergamo, Italy). The source code was written by M.S. Nobile.

Please check out the Wiki for additional descriptions. 

If you need any information about FST-PSO please write to: nobile@disco.unimib.it

FST-PSO requires two packages: miniful and numpy. 

[1] Nobile, Cazzaniga, Besozzi, Colombo, Mauri, Pasi, "Fuzzy Self-Tuning PSO:
A Settings-Free Algorithm for Global Optimization", Swarm & Evolutionary 
Computation, 2017 (doi:10.1016/j.swevo.2017.09.001)

[2] Kennedy, Eberhart, Particle swarm optimization, in: Proceedings IEEE
International Conference on Neural Networks, Vol. 4, 1995, pp. 1942–1948

<http://www.sciencedirect.com/science/article/pii/S2210650216303534>
