from fstpso import FuzzyPSO	

def example_fitness( particle ):
	return sum(map(lambda x: x**2, particle))


if __name__ == '__main__':
	
	dims = 10
	FP = FuzzyPSO()
	FP.set_search_space( [[-10, 10]]*dims )	
	FP.set_fitness(example_fitness)	
	result =  FP.solve_with_fstpso(max_iter=10, max_iter_without_new_global_best=10, verbose=False, save_checkpoint="bla")
	
	print ("Best solution:", result[0])
	print ("Whose fitness is:", result[1])

	FP2 = FuzzyPSO()
	FP2.set_search_space( [[-10, 10]]*dims )	
	FP2.set_fitness(example_fitness)
	result2 =  FP2.solve_with_fstpso(max_iter=100, max_iter_without_new_global_best=10, restart_from_checkpoint="bla")	

	print ("Best solution:", result2[0])
	print ("Whose fitness is:", result2[1])

