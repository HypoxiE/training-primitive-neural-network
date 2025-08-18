import classes

def generate_dataset(N=1000):
	return classes.DataSet(lambda x: 1*x-12, N)

if __name__ == "__main__":
	print([[i.train, i.train_answs] for i in [generate_dataset(10)]])