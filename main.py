
import generate
import visual
import neuro_defs


dataset = generate.generate_dataset(100)


# Создаём и обучаем сеть
nn = neuro_defs.SimpleNN()
nn.train(dataset.train, dataset.train_answs, epochs=100)

# Проверяем на новой точке
for dot in range(len(dataset.test)):
	print(nn.forward(dataset.test[dot]).val, dataset.test_answs[dot])
print()
print(nn.w_out.val, nn.b_out.val)
# visual.plot_dataset(dataset)
# visual.plt_show()