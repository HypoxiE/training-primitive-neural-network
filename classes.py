import copy
import random
from auto_diff import auto_diff

class DataSet:
	def __init__(self, func, N=1000) -> None:
		self.train = []
		self.train_answs = []
		self.test = []
		self.test_answs = []

		def gen_data():
			x1 = random.uniform(-1, 1)
			x2 = random.uniform(-1, 1)
			y = func(x1, x2)
			return (x1, x2), y

		for i in range(N):
			(x1, x2), y = gen_data()
			self.train.append((x1, x2))
			self.train_answs.append(y)

		for i in range(N//5):
			(x1, x2), y = gen_data()
			self.test.append((x1, x2))
			self.test_answs.append(y)


	def __repr__(self):
		return f"test: {self.test}\nansws: {self.test_answs}"
	

class Neuron:
	def __init__(self, weights_num: int) -> None:
		self.weights = [auto_diff.Node(random.uniform(-1, 1)) for _ in range(weights_num)]
		self.b = auto_diff.Node(random.uniform(-1, 1))

	def __repr__(self, debug: bool = False) -> str:
		if not debug:
			return f"<weights = {len(self.weights)} b = {float(self.b)}>"
		else:
			return f"<weights = {self.weights} b = {float(self.b)}>"
		
	def update_weights(self, lr):
		for i in range(len(self.weights)):
			self.weights[i] = auto_diff.update_weights(self.weights[i], lr)

		self.b = auto_diff.update_weights(self.b, lr)

def my_sum(array: list) -> int:
	result = 0
	for i in array:
		result += i
	return result

class NeuronNetwork:
	def __init__(self, neurons_num: int, layers_num: int, inputs_num: int, outputs_num: int = 1) -> None:
		'''
		neurons_num: количество нейронов на каждом слое\n
		layers_num: количество слоёв\n
		inputs_num: количество входных нейронов\n
		outputs_num: количество выходных нейронов (по умолчанию 1)\n
		'''
		neurons = []
		for layer in range(layers_num+1):
			neurons.append([])

			if layer == layers_num:
				for neuron in range(outputs_num):
					neurons[layer].append(Neuron(neurons_num))
				continue

			for neuron in range(neurons_num):
				if layer == 0:
					neurons[layer].append(Neuron(inputs_num))
				else:
					neurons[layer].append(Neuron(neurons_num))
		
		self.neurons = neurons

	def forward(self, func, *args, func_out=None):
		if func_out is None:
			func_out = func


		prev_out = [*args]
		out = []
		for layer in self.neurons[:-1]:
			for neuron in layer:
				out.append(func(my_sum([prev_out[i]*neuron.weights[i] for i in range(len(prev_out))]) + neuron.b))
			prev_out = copy.deepcopy(out)
			out = []

		for neuron in self.neurons[-1]:
			out.append(func_out(my_sum([prev_out[i]*neuron.weights[i] for i in range(len(prev_out))]) + neuron.b))
		
		return out[0]

	def update_weights(self, lr=0.1):
		for i in range(len(self.neurons)):
			for j in range(len(self.neurons[i])):
				self.neurons[i][j].update_weights(lr)

if __name__ == "__main__":
	print(NeuronNetwork(4, 4, 2).neurons)