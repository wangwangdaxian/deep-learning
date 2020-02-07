import numpy as np
from keras import models, layers
from keras.datasets import boston_housing

import utils

# %% 准备数据
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std


# %% 构建网络
def build_model():
    # Because we will need to instantiate the same model multiple times,（因为需要将同一个模型多次实例化，）
    # we use a function to construct it.（所以用一个函数来构建模型）
    a_model = models.Sequential()
    a_model.add(layers.Dense(64, activation='relu',
                             input_shape=(train_data.shape[1],)))
    a_model.add(layers.Dense(64, activation='relu'))
    a_model.add(layers.Dense(1))
    a_model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return a_model


# %% K折验证
k = 4
num_val_samples = len(train_data) // k
num_epochs = 200
all_loss = []
all_val_loss = []
all_mae = []
all_val_mae = []
for i in range(k):
    print('processing fold #', i)
    # Prepare the validation data: data from partition # k（准备验证数据：第 k 个分区的数据）
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

    # Prepare the training data: data from all other partitions（准备训练数据：其他所有分区的数据）
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]],
        axis=0)

    # Build the Keras model (already compiled)（构建 Keras 模型（已编译））
    model = build_model()
    # Train the model (in silent mode, verbose=0)（训练模型（静默模式，verbose=0））
    hist = model.fit(partial_train_data, partial_train_targets,
                     validation_data=(val_data, val_targets),
                     epochs=num_epochs, batch_size=1, verbose=0)
    history = hist.history
    all_loss.append(history['loss'])
    all_val_loss.append(history['val_loss'])
    all_mae.append(history['mae'])
    all_val_mae.append(history['val_mae'])


# %% 绘图
def cal_mean(all_value, epochs):
    return [np.mean([x[j] for x in all_value]) for j in range(epochs)]


def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


ave_loss = smooth_curve(cal_mean(all_loss, num_epochs))
ave_val_loss = smooth_curve(cal_mean(all_val_loss, num_epochs))
ave_mae = smooth_curve(cal_mean(all_mae, num_epochs))
ave_val_mae = smooth_curve(cal_mean(all_val_mae, num_epochs))
utils.draw_loss(ave_loss, ave_val_loss)
utils.draw_mae(ave_mae, ave_val_mae)

# %% 完成模型调参后训练最佳模型
model = build_model()
# Train it on the entirety of the data.
model.fit(train_data, train_targets,
          epochs=80, batch_size=16, verbose=0)
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
