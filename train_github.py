
"""
# File       : train.py
# Time       ：2023/6/19 9:38
# Author     ：Zhang Wenyu
# Description：
"""


import numpy as np

from datetime import datetime
import model
from keras.layers import Dense
from sklearn.metrics import classification_report
import os
import sys
from keras.backend import manual_variable_initialization
from tensorflow.keras.callbacks import EarlyStopping

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
import pandas as pd

manual_variable_initialization(True)
# 核心代码，设置显示的最大列、宽等参数，消掉打印不完全中间的省略号
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)
from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
from sklearn import metrics
# from tf.keras.callbacks import ModelCheckpoint
import keras
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
# from keras_preprocessing.sequence import pad_sequences

import numpy as np
from gensim.models.word2vec import Word2Vec
import gensim

import re

from nltk.stem import WordNetLemmatizer
from keras.callbacks import Callback
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from collections import Counter
# from matplotlib.font_manager import FontProperties  # 字体管理器
import math
import pickle

from typing import Dict, List, Tuple
import getData


# 数据预处理 去除标点符号，词形还原，停用词过滤
def split_camel_case_list(strings):
    result_list = []
    for string in strings:
        # 使用正则表达式匹配驼峰表达式
        matches = re.findall(r'(?:[A-Z]|^)(?:[a-z]+|[A-Z]*(?=[A-Z]|$))', string)
        if matches:
            # 将匹配到的部分连接成一个新的字符串，并添加到结果列表中
            result_list.extend(matches)
        else:
            # 没有匹配到驼峰表达式，直接将字符串加入结果列表
            result_list.append(string)
    return result_list


def lowercase_words(strings):
    result_list = []
    for string in strings:
        # 将字符串中的每个单词都转换为小写并添加到结果列表中
        result_list.append(string.lower())
    return result_list


# 数据预处理 去除标点符号，词形还原，停用词过滤
def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    words = text.split()
    words = lowercase_words(split_camel_case_list(words))
    lemmatizer = WordNetLemmatizer()
    lem_words = [lemmatizer.lemmatize(w, pos='n') for w in words]
    stopwords = {}.fromkeys([line.rstrip() for line in open('model/stopwords.txt')])
    eng_stopwords = set(stopwords)
    words = [w for w in lem_words if w not in eng_stopwords]
    return words


class ClassMetricsCallback(keras.callbacks.Callback):
    def __init__(self, valid_data, test_data, num_classes, class_name):
        self.x_valid, self.y_valid = valid_data
        self.x_test, self.y_test = test_data
        self.num_classes = num_classes
        self.class_name = class_name
        self.best_acc = 0.0
        self.best_model_weights = None

    def on_train_end(self, logs=None):
        """
        Test
        on_train_end _summary_

        _extended_summary_

        Parameters
        ----------
        logs : _type_, optional
            _description_, by default None
        """
        x_val = self.x_test
        y_val = np.argmax(tf.convert_to_tensor(self.y_test), axis=1)
        # 进行预测
        predictions = self.model.predict(x_val)

        # 将预测结果转换为类别标签
        predicted_labels = np.argmax(predictions, axis=1)
        print(f"--------------TEST---------")
        report = classification_report(y_val, predicted_labels, digits=4)
        print(report)

        tnr_by_class = {}
        sample_count_by_class = {}
        # 计算每一类的准确率、精确率和召回率
        for class_label in range(self.num_classes):
            # 获取属于当前类别的样本索引
            class_indices = np.where(y_val == class_label)[0]

            # 获取属于当前类别的预测结果和真实标签
            class_predictions = tf.gather(predicted_labels, class_indices)
            class_true_labels = tf.gather(y_val, class_indices)

            # 计算准确率、精确率和召回率
            accuracy = accuracy_score(class_true_labels, class_predictions)


            class_name = self.class_name[class_label]
            # 输出指标
            print(f"{class_label} is {class_name},Accuracy: {accuracy:.4f}")



    def on_epoch_end(self, epoch, logs=None):
        acc = logs['val_accuracy']
        if acc > self.best_acc:
            self.best_acc = acc
            self.best_model_weights = self.model.get_weights()
        else:
            return
        # 获取验证集数据
        # 获取验证数据
        x_val = self.x_valid
        y_val = np.argmax(tf.convert_to_tensor(self.y_valid), axis=1)
        # 进行预测
        predictions = self.model.predict(x_val)

        # 将预测结果转换为类别标签
        predicted_labels = np.argmax(predictions, axis=1)
        print(f"Epoch {epoch + 1}")
        report = classification_report(y_val, predicted_labels)
        print(report)

        tnr_by_class = {}
        sample_count_by_class = {}
        # 计算每一类的准确率、精确率和召回率
        for class_label in range(self.num_classes):
            # 获取属于当前类别的样本索引
            class_indices = np.where(y_val == class_label)[0]

            # 获取属于当前类别的预测结果和真实标签
            # 获取属于当前类别的预测结果和真实标签
            class_predictions = tf.gather(predicted_labels, class_indices)
            class_true_labels = tf.gather(y_val, class_indices)
            # 计算准确率、精确率和召回率
            accuracy = accuracy_score(class_true_labels, class_predictions)
            # 计算准确率、精确率和召回率
            # 计算每个类的真负样本数量
            true_negatives = np.sum((predicted_labels != class_label) & (y_val != class_label))

            real_negatives = np.sum(y_val != class_label)

            tnr = true_negatives / real_negatives

            # 计算真负率
            # tnr = true_negatives / real_negatives

            # tnr = tn / total_negative
            tnr_by_class[class_label] = tnr
            sample_count_by_class[class_label] = len(class_indices)
            class_name = self.class_name[class_label]
            # 输出指标
            print(f"{class_label} is {class_name},Accuracy: {accuracy:.4f},TNR: {tnr:.4f}")
        # 计算Macro Average TNR
        macro_average_tnr = np.mean(list(tnr_by_class.values()))

        # 计算Weighted TNR
        weighted_tnr = np.sum([tnr_by_class[class_label] * sample_count_by_class[class_label] for class_label in
                               range(self.num_classes)]) / np.sum(list(sample_count_by_class.values()))
        print("Macro Average TNR:", macro_average_tnr)
        print("Weighted TNR:", weighted_tnr)



data_path = 'all_data.pickle'

print("------------------------------------------")
print("TRAIN on " + data_path)
print("------------------------------------------")
# document_list, weights_list, metric_list, sub_list, labels_list, num_class  = getData.read_data(folder_path)
document_list, weights_list, metric_list, sub_list, labels_list, class_name = getData.read_pickle(data_path)
num_class = len(class_name)
# clean the text?
for i in range(len(document_list)):
    texts = document_list[i]
    texts = [clean_text(text) for text in texts]
    document_list[i] = texts
print(len(document_list))

for i in range(len(sub_list)):
    texts = sub_list[i]
    texts = [clean_text(text) for text in texts]
    sub_list[i] = texts

w2v_model = Word2Vec.load('model/middle_word2vec.pkl')


vocab = w2v_model.wv.index2word
# 构建单词到ID的映射关系
vocab = {word: i for i, word in enumerate(vocab)}

# 拟合Tokenizer以构建词汇表
all_docs = []
for doc in document_list:
    all_docs.extend(doc)
# tokenizer.fit_on_texts(all_docs)

# 对每个文档进行处理
padded_sequences = []
max_lines = 100  # 每个文档的最大行数
max_sub_lines = 30  # subgraph的最大行数
max_length = 100
feature_dim = 300

for i in range(len(document_list)):
    doc = document_list[i]
    weights = weights_list[i]
    # 将文档转换为ID序列
    # doc_ids = tokenizer.texts_to_sequences(doc)

    doc_ids = [[vocab.get(word, 0) for word in word_list] for word_list in doc]

    # 截断或补充行数至最大行数
    if len(doc_ids) > max_lines:
        doc_ids = doc_ids[:max_lines]  # 截断多余的行
        weights = weights[:max_lines]
    else:
        while len(doc_ids) < max_lines:
            doc_ids.append([0])  # 补充0直到达到最大行数
            weights.append(0)

    # 对ID序列进行填充
    # 每一行40词
    padded_seqs = pad_sequences(doc_ids, maxlen=max_length, padding='post')
    padded_sequences.append(padded_seqs)

    weights_list[i] = weights

padded_sub_sequences = []
for i in range(len(sub_list)):
    doc = sub_list[i]
    weights = metric_list[i]
    # 将文档转换为ID序列
    # doc_ids = tokenizer.texts_to_sequences(doc)
    doc_ids = [[vocab.get(word, 0) for word in word_list] for word_list in doc]

    # 截断或补充行数至最大行数
    if len(doc_ids) > max_sub_lines:
        doc_ids = doc_ids[:max_sub_lines]  # 截断多余的行
        weights = weights[:max_sub_lines]
    else:
        while len(doc_ids) < max_sub_lines:
            doc_ids.append([0])  # 补充0直到达到最大行数
        """
        这里可能有问题，存疑
        因为两个长度可能不等，初步怀疑是预处理的时候 处理了两遍结果不一样
        """
        while len(weights) < max_sub_lines:
            weights.append(np.zeros(3))

    # 对ID序列进行填充
    # 每一行40词
    padded_seqs = pad_sequences(doc_ids, maxlen=max_length, padding='post')
    padded_sub_sequences.append(padded_seqs)

    metric_list[i] = weights

# vocab = tokenizer.word_index  # 得到每个词的编号


with open('model/keywords_info_gain', 'r') as f:
    key_words_importance = eval(f.read())

# 预训练的词向量中没有出现的词用0向量表示
embedding_matrix = np.zeros((len(vocab) + 1, 300))
for word, i in vocab.items():
    try:
        if word in key_words_importance:
            embedding_vector = np.dot(w2v_model.wv[word], math.exp(key_words_importance[word]))
            # 高版本word2vec不再支持以上代码
            # embedding_vector = np.dot(w2v_model[word], math.exp(key_words_importance[word]))
            embedding_matrix[i] = embedding_vector
        else:
            embedding_vector = w2v_model.wv[str(word)]
            # embedding_vector = w2v_model[str(word)]
            embedding_matrix[i] = embedding_vector
    except KeyError:
        continue

# Create the TextCGRU model
global_model = model.CNN_model(embedding_matrix, vocab, num_class, max_lines, max_length, 1, feature_dim)
Sub_model = model.Sub_model(embedding_matrix, vocab, num_class, max_sub_lines, max_length, 3, feature_dim)

model = model.fusion(global_model, Sub_model, num_class, max_lines, max_sub_lines, max_length, 1)
model.summary()

for i in range(len(weights_list)):
    weights_list[i] = np.array(weights_list[i])

train_documents, test_documents, train_weights, test_weights, train_labels, test_labels, train_sub, test_sub, train_metric, test_metric = train_test_split(
    padded_sequences, weights_list, labels_list, padded_sub_sequences, metric_list, test_size=0.2, random_state=0)

train_documents, valid_documents, train_weights, valid_weights, train_labels, valid_labels, train_sub, valid_sub, train_metric, valid_metric = train_test_split(
    train_documents, train_weights, train_labels, train_sub, train_metric, test_size=0.15, random_state=0)
train_documents = tf.convert_to_tensor(np.asarray(train_documents))
test_documents = tf.convert_to_tensor(np.asarray(test_documents))
valid_documents = tf.convert_to_tensor(np.asarray(valid_documents))
test_weights = tf.convert_to_tensor(np.asarray(test_weights))
train_weights = tf.convert_to_tensor(np.asarray(train_weights))
valid_weights = tf.convert_to_tensor(np.asarray(valid_weights))
test_labels = tf.convert_to_tensor(np.asarray(test_labels))
train_labels = tf.convert_to_tensor(np.asarray(train_labels))
valid_labels = tf.convert_to_tensor(np.asarray(valid_labels))
test_sub = tf.convert_to_tensor(np.asarray(test_sub))
train_sub = tf.convert_to_tensor(np.asarray(train_sub))
valid_sub = tf.convert_to_tensor(np.asarray(valid_sub))

# 逐个将NumPy数组转换为TensorFlow张量，并存储在一个列表中
tensor_list = [tf.convert_to_tensor(arr) for arr in test_metric]
# 使用tf.stack()函数将张量堆叠在一起，创建一个包含所有张量的张量
test_metric = tf.stack(tensor_list)

# 逐个将NumPy数组转换为TensorFlow张量，并存储在一个列表中
tensor_list = [tf.convert_to_tensor(arr) for arr in train_metric]
# 使用tf.stack()函数将张量堆叠在一起，创建一个包含所有张量的张量
train_metric = tf.stack(tensor_list)

tensor_list = [tf.convert_to_tensor(arr) for arr in valid_metric]
# 使用tf.stack()函数将张量堆叠在一起，创建一个包含所有张量的张量
valid_metric = tf.stack(tensor_list)

test_labels = tf.one_hot(test_labels, depth=num_class)
train_labels = tf.one_hot(train_labels, depth=num_class)
valid_labels = tf.one_hot(valid_labels, depth=num_class)

from keras.callbacks import ReduceLROnPlateau

# 创建ReduceLROnPlateau回调函数
lr_scheduler = ReduceLROnPlateau(factor=0.1, patience=3)


checkpoint_path = "./training_1/test.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor='val_accuracy', save_best_only=True,
                                                mode='max', verbose=1)


class PrintTimeCallback(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        print(f"Current time: {datetime.now()}")


# 创建回调函数实例
metrics_callback = ClassMetricsCallback(([valid_documents, valid_weights, valid_sub, valid_metric], valid_labels),
                                        ([test_documents, test_weights, test_sub, test_metric], test_labels),
                                        len(class_name), class_name)
from keras import callbacks

class WeightCheckpoint(callbacks.Callback):
    def __init__(self, filepath, model1, model2):
        super(WeightCheckpoint, self).__init__()
        self.best_acc = 0.0
        self.filepath = filepath
        self.model1 = model1
        self.model2 = model2

    def on_epoch_end(self, epoch, logs=None):
        current_acc = logs['val_accuracy']
        if current_acc > self.best_acc:
            print()
            print(f"improved to {current_acc} from {self.best_acc}, save model to: {self.filepath} ")
            self.best_acc = current_acc
            # self.model.save_weights(self.filepath)
            self.model1.save_weights(self.filepath + 'model1.h5')
            self.model2.save_weights(self.filepath + 'model2.h5')
            self.model.save_weights(self.filepath + 'combined.h5')



dir_path = "0721_trai_pre"
# 回调函数保存最优权重
cp = WeightCheckpoint('./' + dir_path + '/', global_model, Sub_model)
# 'patience' 参数指定在性能不再改善的情况下，需要等待多少个 epoch 才停止训练
os.makedirs(dir_path, exist_ok=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

history = model.fit(
    (train_documents, train_weights, train_sub, train_metric), train_labels,
    batch_size=64,
    epochs=100,
    verbose=1,
    validation_data=((valid_documents, valid_weights, valid_sub, valid_metric), valid_labels),
    shuffle=True,
    callbacks=[early_stopping, lr_scheduler, PrintTimeCallback()]
)

print("over")
