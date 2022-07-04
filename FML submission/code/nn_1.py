import sys
import os
import numpy as np
import pandas as pd

np.random.seed(42)

NUM_FEATS = 90


class Net(object):

	def __init__(self, num_layers, num_units):
		"""
		Initialize the neural network.
		Create weights and biases.

		Parameters
		----------
			num_layers : Number of HIDDEN layers.
			num_units : Number of units in each Hidden layer.
		"""
		self.num_layers = num_layers  # number of hidden layers except output & input layer.
		self.num_units = num_units  # number of neurons in each hidden layer.

		self.layer_inp = [None] * (self.num_layers + 1)
		self.layer_out = [None] * (self.num_layers + 1)
		self.layer_weight = [None] * (self.num_layers + 1)
		self.layer_bias = [None] * (self.num_layers + 1)

		# 0th layer is output layer with single neuron for each layer input to neuron is (batch_size x number of neurons)
		# for 0th layer neuron first column will be input and output since there is only one neuron.
		# top layer has only one neuron
		self.layer_weight[0] = np.random.uniform(-1, 1, self.num_units)
		self.layer_bias[0] = np.random.uniform(-1, 1, 1)
		for layer in range(1, self.num_layers):
			self.layer_weight[layer] = np.random.uniform(-1, 1, (self.num_units, self.num_units))
			self.layer_bias[layer] = np.random.uniform(-1, 1, self.num_units)

		# lowest layer near input
		self.layer_weight[self.num_layers] = np.random.uniform(-1, 1, (NUM_FEATS, self.num_units))
		self.layer_bias[self.num_layers] = np.random.uniform(-1, 1, self.num_units)

		# gradient
		self.delta_layer_inp = [None] * (self.num_layers + 1)
		self.delta_layer_out = [None] * (self.num_layers + 1)
		self.delta_layer_weight = [None] * (self.num_layers + 1)
		self.delta_layer_bias = [None] * (self.num_layers + 1)

	def __call__(self, X):
		"""
		Forward propagate the input X through the network,
		and return the output.

		Parameters
		----------
			X : Input to the network, numpy array of shape m x d
		Returns
		----------
			y : Output of the network, numpy array of shape m x 1
		"""
		#  process last layer i.e. bottom layer near input
		self.layer_inp[self.num_layers] = np.dot(X, self.layer_weight[self.num_layers]) + self.layer_bias[self.num_layers]
		self.layer_out[self.num_layers] = self.relu(self.layer_inp[self.num_layers])

		for layer in range(self.num_layers-1, 0, -1):
			self.layer_inp[layer] = np.dot(self.layer_out[layer+1], self.layer_weight[layer]) + self.layer_bias[layer]
			self.layer_out[layer] = self.relu(self.layer_inp[layer])

		# solve for 0th (output layer) it has single neuron
		self.layer_inp[0] = np.dot(self.layer_out[1], self.layer_weight[0]) + self.layer_bias[0]
		self.layer_out[0] = np.copy(self.layer_inp[0])

		return np.copy(self.layer_out[0])

	def backward(self, X, y, lamda):
		"""
		Compute and return gradients loss with respect to weights and biases.
		(dL/dW and dL/db)

		Parameters
		----------
			X : Input to the network, numpy array of shape m x d
			y : Output of the network, numpy array of shape m x 1
			lamda : Regularization parameter.

		Returns
		----------
			del_W : derivative of loss w.r.t. all weight values (a list of matrices).
			del_b : derivative of loss w.r.t. all bias values (a list of vectors).

		Hint: You need to do a forward pass before performing backward pass.
		"""
		# forward pass is run already just take y_hat
		y_hat = np.copy(self.layer_out[0])
		m = y.shape[0]
		# finding delta for output layer with single neuron f(x) = x.
		self.delta_layer_out[0] = (2 / m) * (y_hat - y)  # for single example loss = (y_hat-y)^2 / 1
		self.delta_layer_inp[0] = np.copy(self.delta_layer_out[0])
		self.delta_layer_weight[0] = np.dot(self.delta_layer_inp[0], self.layer_out[1]) / m
		self.delta_layer_weight[0] += lamda * (2 / m) * self.layer_weight[0]
		self.delta_layer_bias[0] = np.sum(self.delta_layer_inp[0]) / m
		self.delta_layer_bias[0] += lamda * (2 / m) * self.layer_bias[0]

		if self.num_layers == 1:
			# solve for single layer
			# outer multiplication because col_vector x row_vector = matrix (it should be delta matrix input_X_units)
			# instead or using outer we can simply convert vector into col and row matrix using reshape and use np.dot()
			self.delta_layer_out[1] = np.outer(self.delta_layer_inp[0], self.layer_weight[0])
			self.delta_layer_inp[1] = self.delta_layer_out[1] * self.delta_relu(self.layer_inp[1])
			self.delta_layer_weight[1] = np.dot(X.transpose(), self.delta_layer_inp[1]) / m
			self.delta_layer_weight[1] += lamda * (2 / m) * self.layer_weight[1]
			self.delta_layer_bias[1] = np.dot(np.ones(m), self.delta_layer_inp[1]) / m
			self.delta_layer_bias[1] += lamda * (2 / m) * self.layer_bias[1]

		else:
			# solve for multi-layer
			# solve for 1st hidden layer
			self.delta_layer_out[1] = np.outer(self.delta_layer_inp[0], self.layer_weight[0])
			self.delta_layer_inp[1] = self.delta_layer_out[1] * self.delta_relu(self.layer_inp[1])
			self.delta_layer_weight[1] = np.dot(self.layer_out[2].transpose(), self.delta_layer_inp[1]) / m
			self.delta_layer_weight[1] += lamda * (2 / m) * self.layer_weight[1]
			self.delta_layer_bias[1] = np.dot(np.ones(m), self.delta_layer_inp[1]) / m
			self.delta_layer_bias[1] += lamda * (2 / m) * self.layer_bias[1]

			# loop for middle hidden layers
			for layer in range(2, self.num_layers):
				self.delta_layer_out[layer] = np.dot(self.delta_layer_inp[layer-1], self.layer_weight[layer-1].transpose())
				self.delta_layer_inp[layer] = self.delta_layer_out[layer] * self.delta_relu(self.layer_inp[layer])
				self.delta_layer_weight[layer] = np.dot(self.layer_out[layer+1].transpose(), self.delta_layer_inp[layer]) / m
				self.delta_layer_weight[layer] += lamda * (2 / m) * self.layer_weight[layer]
				self.delta_layer_bias[layer] = np.dot(np.ones(m), self.delta_layer_inp[layer]) / m
				self.delta_layer_bias[layer] += lamda * (2 / m) * self.layer_bias[layer]

			# solve for last hidden layer
			layer = self.num_layers
			self.delta_layer_out[layer] = np.dot(self.delta_layer_inp[layer - 1], self.layer_weight[layer - 1].transpose())
			self.delta_layer_inp[layer] = self.delta_layer_out[layer] * self.delta_relu(self.layer_inp[layer])
			self.delta_layer_weight[layer] = np.dot(X.transpose(), self.delta_layer_inp[layer]) / m
			self.delta_layer_weight[layer] += lamda * (2 / m) * self.layer_weight[layer]
			self.delta_layer_bias[layer] = np.dot(np.ones(m), self.delta_layer_inp[layer]) / m
			self.delta_layer_bias[layer] += lamda * (2 / m) * self.layer_bias[layer]

		return self.copy_list_of_np(self.delta_layer_weight), self.copy_list_of_np(self.delta_layer_bias)

	def relu(self, matrix):
		return np.maximum(matrix, 0)

	def delta_relu(self, matrix):
		temp = np.copy(matrix)
		temp[temp <= 0] = 0
		temp[temp > 0] = 1
		return temp

	def copy_list_of_np(self, lst):
		py_list = [None] * len(lst)
		for idx in range(len(lst)):
			py_list[idx] = np.copy(lst[idx])

		return py_list


class Optimizer(object):

	def __init__(self, learning_rate):
		"""
		Create a Stochastic Gradient Descent (SGD) based optimizer with given
		learning rate.

		Other parameters can also be passed to create different types of
		optimizers.

		Hint: You can use the class members to track various states of the
		optimizer.
		"""
		#   add different criteria to object and use those during training.
		self.learning_rate = learning_rate

	def step(self, weights, biases, delta_weights, delta_biases):
		"""
		Parameters
		----------
			weights: Current weights of the network.
			biases: Current biases of the network.
			delta_weights: Gradients of weights with respect to loss.
			delta_biases: Gradients of biases with respect to loss.
		"""
		updated_weights = [None] * len(weights)
		updated_biases = [None] * len(biases)

		for layer in range(len(weights)):
			updated_weights[layer] = np.subtract(weights[layer], self.learning_rate * delta_weights[layer])
			updated_biases[layer] = np.subtract(biases[layer], self.learning_rate * delta_biases[layer])

		return updated_weights, updated_biases


def loss_mse(y, y_hat):
	"""
	Compute Mean Squared Error (MSE) loss between ground-truth and predicted values.

	Parameters
	----------
		y : targets, numpy array of shape m x 1
		y_hat : predictions, numpy array of shape m x 1

	Returns
	----------
		MSE loss between y and y_hat.
	"""

	diff = np.subtract(y_hat, y)
	sqr_error = np.square(diff)
	mse = sqr_error.mean()
	return mse


def loss_regularization(weights, biases):
	"""
	Compute l2 regularization loss.

	Parameters
	----------
		weights and biases of the network.

	Returns
	----------
		l2 regularization loss
	"""
	l2_norm = 0
	for layer in range(len(weights)):
		l2_norm += np.sum(np.square(weights[layer])) + np.sum(np.square(biases[layer]))

	return l2_norm


def loss_fn(y, y_hat, weights, biases, lamda):
	"""
	Compute loss =  loss_mse(..) + lamda * loss_regularization(..)

	Parameters
	----------
		y : targets, numpy array of shape m x 1
		y_hat : predictions, numpy array of shape m x 1
		weights: and
		biases: of the network
		lamda: Regularization parameter

	Returns
	----------
		l2 regularization loss
	"""
	return loss_mse(y, y_hat) + lamda * loss_regularization(weights, biases)


def rmse(y, y_hat):
	"""
	Compute Root Mean Squared Error (RMSE) loss between ground-truth and predicted values.

	Parameters
	----------
		y : targets, numpy array of shape m x 1
		y_hat : predictions, numpy array of shape m x 1

	Returns
	----------
		RMSE between y and y_hat.
	"""
	return np.sqrt(loss_mse(y, y_hat))


def train(
		net, optimizer, lamda, batch_size, max_epochs,
		train_input, train_target,
		dev_input, dev_target
):
	"""
	In this function, you will perform following steps:
		1. Run gradient descent algorithm for `max_epochs` epochs.
		2. For each bach of the training data
			1.1 Compute gradients
			1.2 Update weights and biases using step() of optimizer.
		3. Compute RMSE on dev data after running `max_epochs` epochs.
	"""
	for epoch in range(max_epochs):
		for batch in range(train_input.shape[0] // batch_size):
			train_sample_input = train_input[batch*batch_size: batch*batch_size + batch_size, :]
			train_sample_target = train_target[batch*batch_size: batch*batch_size + batch_size]

			# forward pass
			train_sample_y_hat = net(train_sample_input)

			# backward pass
			delta_weights, delta_biases = net.backward(train_sample_input, train_sample_target, lamda)

			net.layer_weight, net.layer_bias = optimizer.step(net.layer_weight, net.layer_bias, delta_weights, delta_biases)
			# rms_error = rmse(train_sample_y_hat, train_sample_target)
			# print(rms_error)


def get_test_data_predictions(net, inputs):
	"""
	Perform forward pass on test data and get the final predictions that can
	be submitted on Kaggle.
	Write the final predictions to the part2.csv file.

	Parameters
	----------
		net : trained neural network
		inputs : test input, numpy array of shape m x d

	Returns
	----------
		predictions (optional): Predictions obtained from forward pass
								on test data, numpy array of shape m x 1
	"""
	y_hat = net(inputs)
	y_hat = np.insert(y_hat, 0, 454545)
	df = pd.DataFrame(data=y_hat, columns=["Predicted"])
	#  remove the dummy first row
	df = df.iloc[1:]
	df.to_csv('dataset/203050005.csv')

	params = np.zeros(3)
	part_2 = pd.DataFrame(data=params, columns=["Value"])
	part_2.to_csv('dataset/part_2.csv')


def read_data():
	"""
	Read the train, dev, and test datasets
	"""
	# Training data
	cols_names = pd.read_csv('dataset/train.csv', nrows=1).columns.tolist()
	train_input = pd.read_csv('dataset/train.csv', usecols=cols_names[1:])
	train_input = train_input.to_numpy()

	train_target = pd.read_csv('dataset/train.csv').iloc[:, 0]
	train_target = train_target.to_numpy()

	# Dev data
	dev_input = pd.read_csv('dataset/dev.csv', usecols=cols_names[1:])
	dev_input = dev_input.to_numpy()

	dev_target = pd.read_csv('dataset/dev.csv').iloc[:, 0]
	dev_target = dev_target.to_numpy()

	# Test data
	test_input = pd.read_csv('dataset/test.csv')
	test_input = test_input.to_numpy()

	return train_input, train_target, dev_input, dev_target, test_input


def print_rmse(name, net, input_data, target_data):
	y_hat = net(input_data)
	print(name, " rmse: ", rmse(target_data, y_hat))


def main():
	# These parameters should be fixed for Part 1
	max_epochs = 50
	batch_size = 128

	learning_rate = 0.001
	num_layers = 1
	num_units = 64
	lamda = 0  # Regularization Parameter

	train_input, train_target, dev_input, dev_target, test_input = read_data()
	net = Net(num_layers, num_units)
	optimizer = Optimizer(learning_rate)
	train(
		net, optimizer, lamda, batch_size, max_epochs,
		train_input, train_target,
		dev_input, dev_target
	)

	# get_test_data_predictions(net, test_input)
	print_rmse("train", net, train_input, train_target)
	print_rmse("dev", net, dev_input, dev_target)


if __name__ == '__main__':
	main()
