# %% 下载数据
from keras.datasets import reuters

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

# %% 解码
import utils

word_index = reuters.get_word_index()
utils.show_words(word_index, train_data[0])

# %% 准备数据
import numpy as np
import utils

# Our vectorized training data（将训练数据向量化）
x_train = utils.vectorize_sequences(train_data)
# Our vectorized test data（将测试数据向量化）
x_test = utils.vectorize_sequences(test_data)
# 标签向量化
from keras.utils.np_utils import to_categorical

one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)

# %% 构建网络
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# %% 训练网路
x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

hist = model.fit(partial_x_train,
                 partial_y_train,
                 epochs=20,
                 batch_size=512,
                 validation_data=(x_val, y_val))

# %% 绘制损失图表和精度图像
import utils

utils.draw_loss(hist.history['loss'], hist.history['val_loss'])
utils.draw_acc(hist.history['acc'], hist.history['val_acc'])

# %% 根据图示网络8轮后过拟合，重新训练
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(partial_x_train,
          partial_y_train,
          epochs=8,
          batch_size=512,
          validation_data=(x_val, y_val))
results = model.evaluate(x_test, one_hot_test_labels)

# %% 随机精度测试
import copy

test_labels_copy = copy.copy(test_labels)
np.random.shuffle(test_labels_copy)
float(np.sum(np.array(test_labels) == np.array(test_labels_copy))) / len(test_labels)

# %% 在新数据集上预测结果
predictions = model.predict(x_test)
np.argmax(predictions[0])
