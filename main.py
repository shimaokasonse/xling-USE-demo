import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tf_sentencepiece

module_url = "https://tfhub.dev/google/universal-sentence-encoder-xling-many/1"
embed = hub.Module(module_url)


def plot_similarity(labels, features, rotation):
  corr = np.inner(features, features)
  sns.set(font_scale=1.2)
  g = sns.heatmap(
      corr,
      xticklabels=labels,
      yticklabels=labels,
      vmin=0,
      vmax=1,
      cmap="YlOrRd")
  g.set_xticklabels(labels, rotation=rotation)
  g.set_title("Semantic Textual Similarity")
  plt.show()


def run_and_plot(session_, input_tensor_, messages_, encoding_tensor):
  message_embeddings_ = session_.run(
      encoding_tensor, feed_dict={input_tensor_: messages_})
  plot_similarity(messages_, message_embeddings_, 90)


messages = [
    # Smartphones
    "I like my smart phone.",
    "スマホが好きです。",
    "私のスマートフォンは品質が高いです",
    "キミのスマホかっこいいね！",

    # Weather
    "It will rain tomorrow.",
    "明日は雨かな?",
    "台風の季節ですね。",
    "地球温暖化の影響が現れ始めているのかな",

    # Food and health
    "りんごを一日一個食べると医者いらずですよ",
    "イチゴが食べたい",
    "ダイエットに効く食品を教えて?",

    # Asking about age
    "How old are you?",
    "何歳ですか",
    "いくつですか？",
]

similarity_input_placeholder = tf.placeholder(tf.string, shape=(None))
similarity_message_encodings = embed(similarity_input_placeholder)
with tf.Session() as session:
  session.run(tf.global_variables_initializer())
  session.run(tf.tables_initializer())
  run_and_plot(session, similarity_input_placeholder, messages,
               similarity_message_encodings)