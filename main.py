
import generate
import visual
import neuro_defs


dataset = generate.generate_dataset(10_000)


# Создаём и обучаем сеть
nn = neuro_defs.SimpleNN()
nn.train(dataset.train, epochs=10)

# Проверяем на новой точке
for dot in dataset.test[:10]:
	print(nn.forward(dot.x, dot.y), dot.__repr__())

# visual.plot_dataset(dataset)
# visual.plt_show()