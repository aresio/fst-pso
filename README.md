# FST-PSO

*Fuzzy Self-Tuning PSO* (FST-PSO) is a swarm intelligence global optimization method [1]
based on Particle Swarm Optimization [2].

FST-PSO is designed for the optimization of real- or discrete-valued multi-dimensional minimization problems.

FST-PSO is settings-free version of PSO which exploits fuzzy logic to dynamically assign the functioning parameters to each particle in the swarm. Specifically, during each generation, FST-PSO determines the optimal choice for the cognitive factor, the social factor, the inertia value, the minimum velocity, and the maximum velocity. FST-PSO also uses an heuristics to choose the swarm size, so that the user must not select any functioning setting.

In order to use FST-PSO, the programmer must implement a custom fitness function and specify the boundaries of the search space for each dimension. The programmer can optionally specify the maximum number of iterations and the swarm size. When the stopping criterion is met, FST-PSO returns the best fitting solution found, along with its fitness value. In the case of discrete problems, FST-PSO also returns the probability distributions of the underlying generative model.


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
		print("Best solution:", result[0])
		print("Whose fitness is:", result[1])

## Installing FST-PSO

`pip install fst-pso`

## FFT-PSO

*Fuzzy Time Travel PSO* (FFT-PSO) is a variant of FST-PSO that explores different optimization scenarios
starting from the same initial population where only the particle that lead to the best solution found
is randomly changed [3]

FFT-PSO works under the assumption that premature convergence could be prevented by backtracking to the 
beginning of an optimization and removing the particle that was ultimately responsible for leading 
the whole swarm to a stalling condition.

FFT-PSO provides the same interface of FST-PSO, with the only exception in the creation of the object.
Indeed, in the object initializer the programmer can specify the additional paramter alpha, that is,
either the number or iterations (if it is an int) or a percentage (if it is a float) of iterations
after that the swarm is rewinded to the initial state and the particle leading to the stall is randomly 
re-initialized.

## Example

FFT-PSO can be used as follows:

	from fftpso import FFTPSO	
	
	def example_fitness( particle ):
		return sum(map(lambda x: x**2, particle))
		
	if __name__ == '__main__':
		dims = 10
		FP = FFTPSO(alpha=0.01)
		FP.set_search_space( [[-10, 10]]*dims )	
		FP.set_fitness(example_fitness)
		result =  FP.solve_with_fstpso(max_iter=200)
		print("Best solution:", result[0])
		print("Whose fitness is:", result[1])

## Further information

FST-PSO has been created by M.S. Nobile, D. Besozzi, G. Pasi, G. Mauri, 
R. Colombo (University of Milan-Bicocca, Italy), and P. Cazzaniga (University
of Bergamo, Italy). The source code is written and maintained by M.S. Nobile.

Please check out the Wiki for additional descriptions. 

If you need any information about FST-PSO please write to: nobile@disco.unimib.it

FST-PSO requires two packages: miniful and numpy. 

[1] Nobile, Cazzaniga, Besozzi, Colombo, Mauri, Pasi, "Fuzzy Self-Tuning PSO:
A Settings-Free Algorithm for Global Optimization", Swarm & Evolutionary 
Computation, 39:70-85, 2018 (doi:10.1016/j.swevo.2017.09.001)

[2] Kennedy, Eberhart, Particle swarm optimization, in: Proceedings IEEE
International Conference on Neural Networks, Vol. 4, 1995, pp. 1942–1948

[3] Papetti, Tangherloni, Coelho, Besozzi, Cazzaniga, Nobile, "We Are Sending You 
Back... to the Optimum! Fuzzy Time Travel Particle Swarm Optimization",
in: García-Sánchez, P., Hart, E., Thomson, S.L. (eds) Applications of Evolutionary 
Computation. EvoApplications 2025. Lecture Notes in Computer Science, vol 15613 

<http://www.sciencedirect.com/science/article/pii/S2210650216303534>

<https://link.springer.com/chapter/10.1007/978-3-031-90065-5_10>