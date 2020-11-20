
class Checkpoint(object):

	def __init__(self, fstpso_instance = None):
		# print (" * FST-PSO's checkpoint object created")
		if fstpso_instance is not None:
			self._import_data(fstpso_instance)

	def _import_data(self, fstpso_instance):
		from copy import deepcopy
		self._Solutions = deepcopy(fstpso_instance.Solutions)
		self._Iteration = fstpso_instance.Iterations
		self._G = deepcopy(fstpso_instance.G)
		self._W = deepcopy(fstpso_instance.W)

	def save_checkpoint(self, path, verbose=False):
		import pickle
		with open(path, "wb") as f:
			pickle.dump(self, f)
		if verbose: print (" * Checkpoint '%s' (at iteration %d) saved." % (path, self._Iteration))