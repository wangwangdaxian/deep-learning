import matplotlib.pyplot as plt
import numpy as np


def show_image(digit):
    """
    可视化描述图像的张量
    :param digit:图像样本
    """
    plt.clf()
    plt.imshow(digit)
    plt.show()


def show_words(word_index, digit):
    """
    解码语句文本
    :param word_index:单词索引
    :param digit:语句样本
    """
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    # Note that our indices were offset by 3（注 意，索引减去了 3）
    # because 0, 1 and 2 are reserved indices for "padding", "start of sequence", and "unknown".
    # （因为 0、1、2 是 为“padding”（ 填 充 ）、“start of sequence”（序列开始）、“unknown”（未知词）分别保留的索引）
    decoded_words = ' '.join([reverse_word_index.get(i - 3, '?') for i in digit])
    print(decoded_words)


def vectorize_sequences(sequences, dimension=10000):
    """
    向量化序列为2D张量
    :param sequences:序列
    :param dimension:维度大小
    :return: 2D张量
    """
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


def draw_loss(hist):
    """
    绘制损失图
    :param hist:模型训练返回的结果
    """
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.clf()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def draw_acc(hist):
    """
    绘制精度图
    :param hist:模型训练返回的结果
    """
    acc = hist.history['accuracy']
    val_acc = hist.history['val_accuracy']
    epochs = range(1, len(acc) + 1)
    plt.clf()
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
