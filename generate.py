import random
class Dot:
	def __init__(self, x: float, y: float) -> None:
		self.x = x
		self.y = y
		self.classification = float(((x**2 + y**2)**0.5)>=0.5)

	def get_tup(self) -> tuple:
		return (self.x, self.y, self.classification)
	
	def __str__(self) -> str:
		return f"({self.x}, {self.y})"
	
	def __repr__(self) -> str:
		return f"({self.x}, {self.y}, {self.classification})"

class Dataset:
	def __init__(self, train: list[Dot], test: list[Dot]) -> None:
		self.train = train
		self.test = test
	
	def __str__(self) -> str: 
		return f"Train: {str(self.train)}\nTest: {str(self.test)}"
	
	def __repr__(self) -> str: 
		return f"Train: {self.train}\nTest: {self.test}"
	
def generate_data() -> Dot: 
	return Dot(random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0))

def generate_dataset(N = 1000) -> Dataset:
	return Dataset([generate_data() for i in range(N//5*4)], [generate_data() for i in range(N//5)])

if __name__ == "__main__":
	data = generate_dataset(10)
	print(data)