
import generate
import visual
import neuro_defs


dataset = generate.generate_dataset(1000)


# Создаём и обучаем сеть
nn = neuro_defs.SimpleNN()
nn.train(dataset.train, dataset.train_answs, epochs=100)

# Проверяем на новой точке
for dot in dataset.test[:10]:
	print(nn.forward(dot), dot)

# visual.plot_dataset(dataset)
# visual.plt_show()