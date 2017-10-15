import fuzzy
import fuzzy.storage.fcl.Reader

def process_operators(operator, verbose=False):
	if isinstance(operator, fuzzy.operator.Input.Input):
		if verbose: print "    Simple input", operator.adjective.getName(fuzzySystem), 
		MEMB = operator.adjective.membership
		if verbose: print "membership =", MEMB
		if MEMB==None:
			warnings.warn("input for "+operator.adjective.getName(self.fuzzySystem)[1]+" not specified")
		return MEMB
		
	elif isinstance(operator, fuzzy.operator.Compound.Compound):
		if verbose: print "    Multiple inputs: "			
		BEST = sys.float_info.max
		for inp in operator.inputs:
			MEMB = process_operators(inp, verbose=verbose)
			if MEMB==None: 
					warnings.warn("input for "+inp.adjective.getName(self.fuzzySystem)[1]+" not specified")
			elif MEMB<BEST:
				BEST = MEMB			
		return BEST
				


def get_singleton_value(adj):
	return adj.set.getCOG()

def return_all_output_variable_names(fuzzySystem):
	ret = {}
	for name, i in fuzzySystem.variables.items():
		if isinstance(i , fuzzy.OutputVariable.OutputVariable):
			ret[str(name)]=[0, 0]
	return ret

def actual_sugeno_inference(fuzzySystem, verbose=False):

	all_outputs = return_all_output_variable_names(fuzzySystem)		

	for name, rule in fuzzySystem.rules.items():
		if verbose: print " * Processing rule", name

		MEMB = 0
		DENO = 0
		NUME = 0

		SINGLETON = 0


		if isinstance(rule.adjective, fuzzy.Adjective.Adjective):
			if verbose: print "WARNING: Aggettivo output semplice, non supportato"
			pass
		elif isinstance(rule.adjective,list):
			
			if verbose: print "  * Processing list of OUTPUT adjectives..."

			# what output are we updating?
			for adj in rule.adjective:
				outlabel, outname = adj.getName(fuzzySystem)
				
				if verbose: print "   ", outlabel, outname, 
				SINGLETON = get_singleton_value(adj)
				if verbose: print "whose singleton is", SINGLETON

			# process operators (i.e., input nodes)
			MEMB = process_operators(rule.operator, verbose=verbose)
			if MEMB==None:
				# raise Exception("error: input for "+inp.adjective.getName(self.fuzzySystem)[1]+" not specified")
				# warnings.warn("input for "+inp.adjective.getName(self.fuzzySystem)[1]+" not specified")
				if verbose: print "WARNING: cannot calculate MAX membership value"
				pass
			else:
				DENO += MEMB	
				NUME += MEMB*SINGLETON
				if verbose: print "      Product =", MEMB*SINGLETON

		else:
			raise Exception("rule target not set.")

		all_outputs[outname][0]+=NUME
		all_outputs[outname][1]+=DENO

	if verbose: print all_outputs

	final = {}
	for out, valout in all_outputs.items():
		if valout[1]==0:
			final[out]=0
		else:
			final[out] = valout[0]/valout[1]
	return final

def sugeno_inference(my_input, fuzzySystem, verbose=False):
	"""
		Calculates the Sugeno rule of inference using the specified
		input values and updating the specified output values.
		Appearently, the Sugeno method is not implemented in pyfuzzy. 
		This method replaces pyfuzzy's calculate() method.
	"""

	fuzzySystem.reset()
	fuzzySystem.fuzzify(my_input)
	
	# the following replaces pyfuzzy's inference() method
	my_output = actual_sugeno_inference(fuzzySystem, verbose=verbose)

	# self.fuzzySystem.defuzzify(my_output)
	return my_output



if __name__ == '__main__':
	
	pass