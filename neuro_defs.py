import math
import random

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

class SimpleNN:
	def __init__(self):
		# инициализация весов случайными числами
		self.w1 = random.uniform(-1, 1)
		self.w2 = random.uniform(-1, 1)
		self.b = random.uniform(-1, 1)   # смещение
		self.w_out = random.uniform(-1, 1)
		self.b_out = random.uniform(-1, 1)
		self.lr = 0.001   # скорость обучения

	def forward(self, x1, x2):
		# прямой проход
		self.z1 = self.w1 * x1 + self.w2 * x2 + self.b
		self.a1 = sigmoid(self.z1)  # активация скрытого слоя
		self.z2 = self.w_out * self.a1 + self.b_out
		self.a2 = sigmoid(self.z2)  # выход сети
		return self.a2

	def backward(self, x1, x2, y):
		# вычисляем ошибку
		error = self.a2 - y  # dL/da2

		# производные для выходного слоя
		d_out = error * sigmoid_derivative(self.z2)
		self.w_out -= self.lr * d_out * self.a1
		self.b_out -= self.lr * d_out

		# производные для скрытого слоя
		d_hidden = d_out * self.w_out * sigmoid_derivative(self.z1)
		self.w1 -= self.lr * d_hidden * x1
		self.w2 -= self.lr * d_hidden * x2
		self.b -= self.lr * d_hidden

	def train(self, data, epochs=1000):
		for _ in range(epochs):
			for x1, x2, y in [i.get_tup() for i in data]:
				self.forward(x1, x2)
				self.backward(x1, x2, y)

