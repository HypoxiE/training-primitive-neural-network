import random

class DataSet:
	def __init__(self, N=1000) -> None:
		self.train = []
		self.train_answs = []
		self.test = []
		self.test_answs = []

		for i in range(N//5*4):
			x = random.uniform(-1000, 1000)
			self.train.append(x)
			self.train_answs.append(x+1)

		for i in range(N//5):
			x = random.uniform(-1000, 1000)
			self.test.append(x)
			self.test_answs.append(x+1)