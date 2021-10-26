# This example is based on the code written by pyswarms' authors:
# https://pyswarms.readthedocs.io/en/latest/examples/usecases/train_neural_network.html
#

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from fstpso import FuzzyPSO

def f(x, arguments):
	"""Higher-level method to do forward_prop in the
	whole swarm.

	Inputs
	------
	x: numpy.ndarray of shape (n_particles, dimensions)
		The swarm that will perform the search

	Returns
	-------
	numpy.ndarray of shape (n_particles, )
		The computed loss for each particle
	"""
	x = np.array(x)
	n_particles = x.shape[0]
	j = [forward_prop(x[i], arguments=arguments) for i in range(n_particles)]
	return np.array(j)

# Forward propagation
def forward_prop(params, arguments):
	"""Forward propagation as objective function

	This computes for the forward propagation of the neural network, as
	well as the loss. It receives a set of parameters that must be
	rolled-back into the corresponding weights and biases.

	Inputs
	------
	params: np.ndarray
		The dimensions should include an unrolled version of the
		weights and biases.

	Returns
	-------
	float
		The computed negative log-likelihood loss given the parameters
	"""
	# Neural network architecture
	n_inputs = 4
	n_hidden = 20
	n_classes = 3

	# Roll-back the weights and biases
	W1 = params[0:80].reshape((n_inputs,n_hidden))
	b1 = params[80:100].reshape((n_hidden,))
	W2 = params[100:160].reshape((n_hidden,n_classes))
	b2 = params[160:163].reshape((n_classes,))

	# Perform forward propagation
	z1 = arguments["x"].dot(W1) + b1  # Pre-activation in Layer 1
	a1 = np.tanh(z1)	 # Activation in Layer 1
	z2 = a1.dot(W2) + b2 # Pre-activation in Layer 2
	logits = z2		  # Logits for Layer 2

	# Compute for the softmax of the logits
	exp_scores = np.exp(logits)
	probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

	# Compute for the negative log likelihood
	N = 150 # Number of samples
	corect_logprobs = -np.log(probs[range(N), arguments["y"]])
	loss = np.sum(corect_logprobs) / N

	return loss

def predict(X, pos):
    """
    Use the trained weights to perform class predictions.

    Inputs
    ------
    X: numpy.ndarray
        Input Iris dataset
    pos: numpy.ndarray
        Position matrix found by the swarm. Will be rolled
        into weights and biases.
    """
    # Neural network architecture
    n_inputs = 4
    n_hidden = 20
    n_classes = 3

    # Roll-back the weights and biases
    W1 = pos[0:80].reshape((n_inputs,n_hidden))
    b1 = pos[80:100].reshape((n_hidden,))
    W2 = pos[100:160].reshape((n_hidden,n_classes))
    b2 = pos[160:163].reshape((n_classes,))

    # Perform forward propagation
    z1 = X.dot(W1) + b1  # Pre-activation in Layer 1
    a1 = np.tanh(z1)     # Activation in Layer 1
    z2 = a1.dot(W2) + b2 # Pre-activation in Layer 2
    logits = z2          # Logits for Layer 2

    y_pred = np.argmax(logits, axis=1)
    return y_pred


if __name__ == '__main__':
	
	# Load the iris dataset
	data = load_iris()

	# Store the features as X and the labels as y
	X = data.data
	y = data.target

	dimensions = (4 * 20) + (20 * 3) + 20 + 3
	
	FSTPSO = FuzzyPSO()
	FSTPSO.set_search_space([[-5,5]]*dimensions)
	FSTPSO.set_parallel_fitness(f, arguments={'x': data.data, 'y': data.target }, skip_test=True)
	FSTPSO.set_swarm_size(100)
	bestpos, bestf = FSTPSO.solve_with_fstpso(max_iter=1000)
	accuracy = (predict(data.data, np.array(bestpos.X))==data.target).mean()
	print("Accuracy: %.3f" % (100*accuracy))
