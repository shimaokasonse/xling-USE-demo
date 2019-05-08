import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
import keras.layers as layers
from keras.models import Model
from keras import backend as K
import numpy as np
import tf_sentencepiece
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

np.random.seed(420)

train_df = pd.read_csv("dataset/fine-grained/train/train.csv")

le = LabelEncoder()
le.fit(train_df.label.values)
n_classes = len(list(le.classes_))

def build_input_fn(df, label_key, num_epochs, shuffle, batch_size):
    def input_fn():
        label = le.transform(df[label_key])
        ds = tf.data.Dataset.from_tensor_slices((dict(df),label))
        if shuffle:
            ds = ds.shuffle(10000)
        ds = ds.batch(batch_size).repeat(num_epochs)
        return ds
    return input_fn


train_input_fn = build_input_fn(train_df,"label",30,True,512)

embedded_text_feature_column = hub.text_embedding_column(
    key="text", 
    module_spec="https://tfhub.dev/google/universal-sentence-encoder-xling-many/1",
    trainable=False)

estimator = tf.estimator.DNNClassifier(
    hidden_units=[512,256],
    feature_columns=[embedded_text_feature_column],
    n_classes=n_classes,
    dropout=0.5,
    optimizer=tf.train.AdagradOptimizer(learning_rate=0.005))


estimator.train(input_fn=train_input_fn)
train_input_fn = build_input_fn(train_df,"label",1,False,128)
print("train accuracy", estimator.evaluate(input_fn=train_input_fn))
p = Path("dataset/fine-grained/test")
for test_file in p.glob("*.csv"):
    test_df = pd.read_csv(test_file)
    test_input_fn = build_input_fn(test_df,"label",1,False,128)
    print("test accuracy {}:".format(test_file),estimator.evaluate(input_fn=test_input_fn))
    predictions = estimator.predict(input_fn=test_input_fn)
    predicted_labels = le.inverse_transform([np.argmax(each['logits']) for each in predictions])
    test_df["predicted"] = predicted_labels
    test_df[["label","predicted","text"]].to_csv("dataset/fine-grained/predict/{}".format(test_file.name),index=False)