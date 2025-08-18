
import generate
import visual
import neuro_defs


dataset = generate.generate_dataset(100)


# Создаём и обучаем сеть
nn = neuro_defs.SimpleNN()
epoch = 100
for i in range(epoch):
	nn.train(dataset.train, dataset.train_answs, epochs=1)

	if epoch % 10 == 0:
		print("*"*(i//10) + "-"*((epoch-i)//10))

# Проверяем на новой точке
for dot in range(len(dataset.test)):
	print(nn.forward(*dataset.test[dot]).val, dataset.test_answs[dot])
print()
# visual.plot_dataset(dataset)
# visual.plt_show()