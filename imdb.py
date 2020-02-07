# %% 下载数据
from keras.datasets import imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# %% 解码
# word_index is a dictionary mapping words to an integer index（word_index 是一个将单词映射为整数索引的字典）
import utils

word_index = imdb.get_word_index()
utils.show_words(word_index, train_data[0])

# %% 准备数据
import utils

# Our vectorized training data（将训练数据向量化）
x_train = utils.vectorize_sequences(train_data)
# Our vectorized test data（将测试数据向量化）
x_test = utils.vectorize_sequences(test_data)
# Our vectorized labels（标签向量化）
from keras.utils.np_utils import to_categorical

y_train = to_categorical(train_labels)
y_test = to_categorical(test_labels)

# %% 构建网络
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
# %% 训练模型
x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

hist = model.fit(partial_x_train,
                 partial_y_train,
                 epochs=20,
                 batch_size=512,
                 validation_data=(x_val, y_val))

# %% 绘制损失图表和精度图像
import utils

utils.draw_loss(hist)
utils.draw_acc(hist)

# %% 上述训练结果出现过拟合 适当减少训练次数
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=4, batch_size=512)
results = model.evaluate(x_test, y_test)

# %% 使用训练好的模型预测结果
model.predict(x_test)
