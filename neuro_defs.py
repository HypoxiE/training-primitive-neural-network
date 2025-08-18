import math
import random
from auto_diff import auto_diff

def func_active(x):
	return

class SimpleNN:
	def __init__(self):
		self.w_out = auto_diff.Node(random.uniform(-1, 1), label="w_out")
		self.b_out = auto_diff.Node(random.uniform(-1, 1), label="b_out")
		self.lr = 0.02   # скорость обучения

	def forward(self, x):
		# прямой проход
		self.z1 = self.w_out * x + self.b_out
		return self.z1

	def backward(self, x, y):
		# вычисляем ошибку
		error = (self.z1 - y)**2  # dL/da2

		auto_diff.backward(error)
		self.w_out = auto_diff.Node(float(self.w_out) - self.lr * self.w_out.grad, label="w_out")
		self.b_out = auto_diff.Node(float(self.b_out) - self.lr * self.b_out.grad, label="b_out")

	def train(self, dataset, answs, epochs=1000):
		for _ in range(epochs):
			for i in range(len(dataset)):
				self.forward(dataset[i])
				self.backward(dataset[i], answs[i])

