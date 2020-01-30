# =================================================
# 该文件不能命名为'numbers.py'，会与内部模块名冲突导致异常
# =================================================
from keras import layers
from keras import models
from keras.datasets import mnist
from keras.utils import to_categorical

# %% 网络架构
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))
# %% 编译步骤
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
# %% 准备图像数据
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255
# %% 准备标签
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
# %% 开始训练
network.fit(train_images, train_labels, epochs=5, batch_size=128)
# %% 测试结果
test_loss, test_acc = network.evaluate(test_images, test_labels)
print(f'test_loss:{test_loss},test_acc:{test_acc}')
