
import random
from auto_diff import auto_diff
import classes

def sigmoid(x: auto_diff.Node):
    return 1 / (1 + (-x).exp())

def tanh(x: auto_diff.Node):
	return (x.exp() - (-x).exp()) / (x.exp() + (-x).exp())

class SimpleNN:
	def __init__(self):
		#111 - 1 слой, 1 нейрон, 1 вес
		self.w111 = auto_diff.Node(random.uniform(-1, 1))
		self.w112 = auto_diff.Node(random.uniform(-1, 1))
		self.b11 = auto_diff.Node(random.uniform(-1, 1))

		self.w121 = auto_diff.Node(random.uniform(-1, 1))
		self.w122 = auto_diff.Node(random.uniform(-1, 1))
		self.b12 = auto_diff.Node(random.uniform(-1, 1))

		self.w131 = auto_diff.Node(random.uniform(-1, 1))
		self.w132 = auto_diff.Node(random.uniform(-1, 1))
		self.b13 = auto_diff.Node(random.uniform(-1, 1))

		self.w141 = auto_diff.Node(random.uniform(-1, 1))
		self.w142 = auto_diff.Node(random.uniform(-1, 1))
		self.b14 = auto_diff.Node(random.uniform(-1, 1))



		self.w211 = auto_diff.Node(random.uniform(-1, 1))
		self.w212 = auto_diff.Node(random.uniform(-1, 1))
		self.w213 = auto_diff.Node(random.uniform(-1, 1))
		self.w214 = auto_diff.Node(random.uniform(-1, 1))
		self.b21 = auto_diff.Node(random.uniform(-1, 1))

		self.w221 = auto_diff.Node(random.uniform(-1, 1))
		self.w222 = auto_diff.Node(random.uniform(-1, 1))
		self.w223 = auto_diff.Node(random.uniform(-1, 1))
		self.w224 = auto_diff.Node(random.uniform(-1, 1))
		self.b22 = auto_diff.Node(random.uniform(-1, 1))



		self.w1_out = auto_diff.Node(random.uniform(-1, 1))
		self.w2_out = auto_diff.Node(random.uniform(-1, 1))
		self.b_out = auto_diff.Node(random.uniform(-1, 1))

		self.lr = 0.1   # скорость обучения

	def forward(self, x1, x2):
		# прямой проход

		#скрытые слои
		self.h11 = tanh(self.w111*x1 + self.w112*x2 + self.b11)
		self.h12 = tanh(self.w121*x1 + self.w122*x2 + self.b12)
		self.h13 = tanh(self.w131*x1 + self.w132*x2 + self.b13)
		self.h14 = tanh(self.w141*x1 + self.w142*x2 + self.b14)

		self.h21 = tanh(self.w211*self.h11 + self.w212*self.h12 + self.w213*self.h13 + self.w214*self.h14 + self.b21)
		self.h22 = tanh(self.w221*self.h11 + self.w222*self.h12 + self.w223*self.h13 + self.w224*self.h14 + self.b22)

		#выходной слой
		self.a1 = sigmoid(self.w1_out*self.h21 + self.w2_out*self.h22 + self.b_out)

		return self.a1

	def backward(self, y):
		# вычисляем ошибку
		error = (self.a1 - y)**2  # dL/da2

		auto_diff.backward(error)

		self.w111 = auto_diff.update_weights(self.w111, self.lr)
		self.w112 = auto_diff.update_weights(self.w112, self.lr)
		self.b11 = auto_diff.update_weights(self.b11, self.lr)

		self.w121 = auto_diff.update_weights(self.w121, self.lr)
		self.w122 = auto_diff.update_weights(self.w122, self.lr)
		self.b12 = auto_diff.update_weights(self.b12, self.lr)

		self.w131 = auto_diff.update_weights(self.w131, self.lr)
		self.w132 = auto_diff.update_weights(self.w132, self.lr)
		self.b13 = auto_diff.update_weights(self.b13, self.lr)

		self.w141 = auto_diff.update_weights(self.w141, self.lr)
		self.w142 = auto_diff.update_weights(self.w142, self.lr)
		self.b14 = auto_diff.update_weights(self.b14, self.lr)



		self.w211 = auto_diff.update_weights(self.w211, self.lr)
		self.w212 = auto_diff.update_weights(self.w212, self.lr)
		self.w213 = auto_diff.update_weights(self.w213, self.lr)
		self.w214 = auto_diff.update_weights(self.w214, self.lr)
		self.b21 = auto_diff.update_weights(self.b21, self.lr)

		self.w221 = auto_diff.update_weights(self.w221, self.lr)
		self.w222 = auto_diff.update_weights(self.w222, self.lr)
		self.w223 = auto_diff.update_weights(self.w223, self.lr)
		self.w224 = auto_diff.update_weights(self.w224, self.lr)
		self.b22 = auto_diff.update_weights(self.b22, self.lr)



		self.w1_out = auto_diff.update_weights(self.w1_out, self.lr)
		self.w2_out = auto_diff.update_weights(self.w2_out, self.lr)
		self.b_out = auto_diff.update_weights(self.b_out, self.lr)

	def train(self, dataset, answs, epochs=1000):
		for _ in range(epochs):
			for i in range(len(dataset)):
				self.forward(*dataset[i])
				self.backward(answs[i])

