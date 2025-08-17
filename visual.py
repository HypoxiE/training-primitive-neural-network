import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


def plot_dataset(dataset):
	x0 = [dot.x for dot in dataset.train if not dot.classification]
	y0 = [dot.y for dot in dataset.train if not dot.classification]
	x1 = [dot.x for dot in dataset.train if dot.classification]
	y1 = [dot.y for dot in dataset.train if dot.classification]

	plt.scatter(x0, y0, color='green', label='Class 0')
	plt.scatter(x1, y1, color='red', label='Class 1')
    
def plot_decision_surface(network, resolution=0.02):
	x_min, x_max = -1, 1
	y_min, y_max = -1, 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
							np.arange(y_min, y_max, resolution))

	# прогоняем сетку через сеть
	Z = np.array([network.predict([x, y]) for x, y in zip(xx.ravel(), yy.ravel())])
	Z = Z.reshape(xx.shape)

	# закрашиваем фон по вероятности
	plt.contourf(xx, yy, Z, levels=50, cmap='RdYlGn', alpha=0.3)
    
def plot_all(dataset, network):
	plt.figure(figsize=(6,6))
	plot_decision_surface(network)
	plot_dataset(dataset)
	plt.xlim(-1, 1)
	plt.ylim(-1, 1)
	plt.legend()

def plt_show():
	plt.show()


if __name__ == "__main__":
	pass