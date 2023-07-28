# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : model.py
# Time       ：2023/6/12 15:00
# Author     ：Zhang Wenyu
# Description：
"""
# from keras import Model
import  keras
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, concatenate, AveragePooling1D
from keras.callbacks import Callback
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from keras.layers import Input, Conv1D, MaxPooling1D, Flatten, LSTM, Dense, Embedding,Conv2D,MaxPooling2D,Lambda,multiply,Concatenate
from keras.layers import Reshape
from keras.backend import stack
import  tensorflow as tf


def Sub_model(embedding_matrix, vocab, num_classes,n_rows,sequence_length,num_weights, feature_dim):
    # 定义输入的维度
    embedding_dim = 300

    # 定义CNN参数
    filters = 128
    kernel_sizes = [3, 5, 4]  # 使用3和5两个不同大小的卷积核


    # 定义词汇表大小和词嵌入矩阵
    vocab_size = len(vocab) + 1

    # 定义输入层
    document_input = Input(shape=(n_rows, sequence_length))
    weights_input = Input(shape=(n_rows, num_weights))

    reshaped_input = Reshape((n_rows * sequence_length,))(
        document_input)  # Reshape input to (batch_size, n_rows * m_length)

    embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim, weights=[embedding_matrix],
                                trainable=False)(reshaped_input)

    embedding_layer = Reshape((n_rows, sequence_length, embedding_dim))(
        embedding_layer)  # Reshape back to (batch_size, n_rows, m_length, embedding_dim)

    conv_layers = []
    for kernel_size in kernel_sizes:
        conv = Conv1D(filters=filters, kernel_size=kernel_size, activation='relu')
        conv_layers.append(conv)

    # CNN层
    conv_outputs_channel1 = []
    conv_outputs_channel2 = []
    conv_outputs_channel3 = []
    for i in range(n_rows):
        row_input = embedding_layer[:, i, :, :]
        row_outputs = []
        for conv in conv_layers:
            conv_output = conv(row_input)
            pool = GlobalMaxPooling1D()(conv_output)
            # pool_flatten = Flatten()(pool)
            row_outputs.append(pool)

        merged_row = concatenate(row_outputs, axis=1)
        # 乘以权重
        conv_outputs_channel1.append(merged_row)

    # 将 conv_outputs 转换为三维张量
    stacked_tensor = tf.stack(conv_outputs_channel1)
    stacked_tensor = tf.transpose(stacked_tensor , perm=[1, 0, 2])
    # 将系数张量的维度转置为 (None, 3, 30)
    transposed_coefficients_tensor = tf.transpose(weights_input, perm=[0, 2, 1])
    # 维度解释：将系数张量的第1维和第2维交换，得到形状为 (None, 3, 30) 的张量。

    # 扩展系数张量的维度
    expanded_coefficients_tensor = tf.expand_dims(transposed_coefficients_tensor, axis=-1)
    # 维度解释：在系数张量的最后一个维度上添加一个维度，得到形状为 (None, 3, 30, 1) 的张量。

    # 将系数乘到原始张量中
    three_channel_tensor = tf.expand_dims(stacked_tensor , axis=1) * expanded_coefficients_tensor
    # 维度解释：在原始张量的第2个维度上添加一个维度，得到形状为 (None, 1, 30, 384) 的张量，然后与扩展后的系数张量逐元素相乘。
    # (None, 3, 30, 384)

    # 转换输入张量的维度顺序
    transposed_tensor = tf.transpose(three_channel_tensor, perm=(0, 2, 3, 1))
    # (None,  30, 384,3)
    # return  stacked_tensor
    # 第一个CNN块
    conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(transposed_tensor )
    norm1 = tf.keras.layers.BatchNormalization()(conv1)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(1, 8))(norm1)
    dropout1 = tf.keras.layers.Dropout(0.25)(pool1)

    # 第二个CNN块
    conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(dropout1)
    norm2 = tf.keras.layers.BatchNormalization()(conv2)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(1, 2))(norm2)
    dropout2 = tf.keras.layers.Dropout(0.25)(pool2)

    # 第三个CNN块
    conv3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')(dropout2)
    norm3 = tf.keras.layers.BatchNormalization()(conv3)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3))(norm3)
    dropout3 = tf.keras.layers.Dropout(0.25)(pool3)

    # 展平层
    flatten = tf.keras.layers.Flatten()(dropout3)

    # 全连接层
    dense = tf.keras.layers.Dense(feature_dim, activation='relu')(flatten)

    # 定义模型
    model = tf.keras.Model(inputs=[document_input, weights_input], outputs=dense)
    return model


def fusion (global_model, sub_model, num_classes,n_rows,sub_rows, sequence_length,num_weights):
    # 定义输入层和模型1
    document_input = Input(shape=(n_rows, sequence_length))
    weights_input = Input(shape=(n_rows, num_weights))
    sub_input = Input(shape=(sub_rows, sequence_length))
    metric_input = Input(shape=(sub_rows, 3))

    model1_output = global_model(inputs=[document_input, weights_input])

    model2_output = sub_model(inputs=[sub_input, metric_input ])

    # 连接两个模型的输出
    concatenated_output = Concatenate()([model1_output, model2_output])
    dense_layer = Dense(200 , activation='relu')(concatenated_output)
    # 添加全连接层进行多分类
    dense_layer = Dense(num_classes, activation='softmax')(dense_layer)

    # 创建最终的模型
    final_model = tf.keras.Model(inputs=[document_input, weights_input,sub_input,metric_input ], outputs=dense_layer)

    # 编译模型
    final_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(),tf.keras.metrics.TrueNegatives(),tf.keras.metrics.FalsePositives()])
    return final_model


def CNN_model(embedding_matrix, vocab, num_classes,n_rows,sequence_length, num_weights, feature_dim):
    """
    CNN MODEL, 传入n行，m个词
    先把每行的词序列转化为embedding
    然后每行学一个CNN的句向量

    :param embedding_matrix:
    :param vocab:
    :param num_classes:
    :param num_weights: global有几个权重
    :return:
    """
    # 定义输入的维度
    embedding_dim = 300

    # 定义CNN参数
    filters = 128
    kernel_sizes = [3, 5, 4]  # 使用3和5两个不同大小的卷积核

    # 定义LSTM参数
    lstm_units = 64

    # 定义词汇表大小和词嵌入矩阵
    vocab_size = len(vocab) + 1


    # 定义输入层
    document_input = Input(shape=(n_rows, sequence_length))
    weights_input = Input(shape=(n_rows, num_weights))

    reshaped_input = Reshape((n_rows * sequence_length,))(document_input)  # Reshape input to (batch_size, n_rows * m_length)

    embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim, weights=[embedding_matrix],
                                trainable=False)(reshaped_input)

    embedding_layer = Reshape((n_rows, sequence_length, embedding_dim))(
        embedding_layer)  # Reshape back to (batch_size, n_rows, m_length, embedding_dim)

    conv_layers = []
    for kernel_size in kernel_sizes:
        conv = Conv1D(filters=filters, kernel_size=kernel_size, activation='relu')
        conv_layers.append(conv)

    # CNN层
    conv_outputs = []
    for i in range(n_rows):
        row_input = embedding_layer[:, i, :, :]
        row_outputs = []
        for conv in conv_layers:
            conv_output = conv(row_input)
            pool = GlobalMaxPooling1D()(conv_output)
            # pool_flatten = Flatten()(pool)
            row_outputs.append(pool )

        merged_row = concatenate(row_outputs, axis=1)
        # 乘以权重
        weighted_row = multiply([merged_row, weights_input[i]])
        # 没加乘的
        conv_outputs.append(weighted_row)

    # 将 conv_outputs 转换为三维张量
    conv_outputs_tensor = stack(conv_outputs)  # 转换为二维张量

    # 定义转换函数
    def stack_and_reshape(x):
        x = tf.stack(x)  # 形状为 (15, None, 384)
        x = tf.transpose(x, perm=[1, 0, 2])  # 转置为 (None, 15, 384)
        return x

    # 使用 Lambda 层进行转换
    x = Lambda(stack_and_reshape)(conv_outputs)  # 形状为 (batch_size, 15, 384)

    # # 将输入张量的维度扩展为 (15, None, 384, 1)
    reshaped_inputs = tf.expand_dims(x, axis=-1)

    """
    2D CNN
    """
    # 定义分类器
    x = Conv2D(filters=32, kernel_size=(2, 2), activation='relu', padding='same')( reshaped_inputs )
    x = MaxPooling2D(pool_size=(1, 8))(x)

    # 添加更多的卷积层和池化层（可选）
    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 7))(x)

    x = Flatten()(x)
    dense = Dense(units=feature_dim, activation='relu')(x)

    # 定义模型
    model = tf.keras.Model(inputs=[document_input, weights_input], outputs=dense)
    return model
