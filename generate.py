import classes

def generate_dataset(N=100):
	return classes.DataSet(lambda x, y: float(((x**2 + y**2)**0.5)>=0.5), N)

if __name__ == "__main__":
	print(generate_dataset(10))