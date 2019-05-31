'''
    @Author:    enginning
    @Date:      2019/05/28
    @Contest:   https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
'''


import os
import numpy as np
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.layers import Dense, Input, LSTM, Bidirectional, Activation, Conv1D, GRU
from keras.callbacks import Callback
from keras.layers import Dropout, Embedding, GlobalMaxPooling1D, MaxPooling1D, Add, Flatten
from keras.preprocessing import text, sequence
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils.vis_utils import plot_model
from keras.models import Model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
print(os.getcwd())
print(os.listdir("../input"))


# Input data files are available in the "../input/" directory.
EMBEDDING_FILE = '../input/glove840b300dtxt/glove.840B.300d.txt'
train = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/train.csv')
test  = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/test.csv')


# 数据填充
train["comment_text"].fillna("fillna")
test["comment_text"].fillna("fillna")
X_train = train["comment_text"].str.lower()
y_train = train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
X_test  = test["comment_text"].str.lower()


# Super parameter
max_features = 100000
maxlen = 150
embed_size = 300
batch_size = 128
epochs = 4


'''
    文本预处理:
        num_words:              需要保留的最大词数，基于词频，便于后面向量化
        fit_on_texts(texts):    texts 为要用以训练的文本列表，统计词频，给每个词分配index，形成字典等等
        texts_to_sequences:     列表序列化，返回列表中每个序列对应于一段输入文本
        sequence.pad_sequences: 将多个序列截断或补齐为相同长度，默认前端补齐、前端截断
'''
tok = text.Tokenizer(num_words=max_features, lower=True)
tok.fit_on_texts(list(X_train)+list(X_test))
X_train = tok.texts_to_sequences(X_train)
X_test = tok.texts_to_sequences(X_test)
x_train = sequence.pad_sequences(X_train, maxlen=maxlen)
x_test = sequence.pad_sequences(X_test, maxlen=maxlen)


'''
    词嵌入: 一般用矩阵分解方法、基于浅窗口的方法
    glove:  词向量工具 (Global Vectors for Word Representation)
    glove.840B.300d.txt 是词典文件，包含词及词对应的词向量，根据词汇的共现信息，将词汇编码成一个向量
'''
embeddings_index = {}
# rstrip(): 删除 string 字符串末尾的指定字符 (默认为空格)
with open(EMBEDDING_FILE, encoding='utf8') as f:
    for line in f:
        values = line.rstrip().rsplit(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs


word_index = tok.word_index     # 获取index和词的对照字典 (所有词，包含低频词的)
# prepare embedding matrix
num_words = min(max_features, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, embed_size))
for word, i in word_index.items():
    if i >= max_features:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


# Build model
sequence_input = Input(shape=(maxlen, ))        # 返回一个张量
x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(sequence_input)
x = SpatialDropout1D(0.2)(x)
x = Bidirectional(GRU(128, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
x = Conv1D(64, kernel_size=3, padding="valid", kernel_initializer="glorot_uniform")(x)
avg_pool = GlobalAveragePooling1D()(x)
max_pool = GlobalMaxPooling1D()(x)
x = concatenate([avg_pool, max_pool])
preds = Dense(6, activation="sigmoid")(x)
model = Model(sequence_input, preds)
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-3), metrics=['accuracy'])
plot_model(model, to_file='Network.svg', show_shapes=True)
# train_test_split: 将样本数据切分为训练集和测试集 (交叉验证)
X_tra, X_val, y_tra, y_val = train_test_split(x_train, y_train, train_size=0.9, random_state=233)


class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: {:d} - score: {:.6f}".format(epoch + 1, score))


'''
    # Save model
    save_best_only=True:    最佳模型不会被覆盖
    EarlyStopping():        当被监测的数据不再提升，则停止训练；其中 patience为没有进步的训练轮数，在这之后训练会被停止
'''
filepath = "weights_base.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max') 
early = EarlyStopping(monitor="val_acc", mode="max", patience=5, restore_best_weights=True)
ra_val = RocAucEvaluation(validation_data=(X_val, y_val), interval=1)
callbacks_list = [ra_val, checkpoint, early]


# Train
model.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs,
          validation_data=(X_val, y_val), callbacks=callbacks_list, verbose=1)

# Test
model.load_weights(filepath)
print('Predicting....')
y_pred = model.predict(x_test, batch_size=1024, verbose=1)


# Submission
submission = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv')
submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = y_pred
submission.to_csv('submission.csv', index=False)