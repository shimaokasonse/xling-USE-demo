# xling-USE-demo

# 多言語Universal Sentence Encoderで言語間転移学習を試す

Universal Sentence EncoderはTransformerを自然言語処理の様々なデータセットを使ってマルチタスク学習させて得られた文表現ベクトルのエンコーダーです。今回使うのは多言語バージョンで、英語、フランス語、ドイツ語、スペイン語、イタリア語、中国語、韓国語、日本語のタスクで学習されたものです。意味が似ている文ならば、それがどの言語であるかによらず、似たようなベクトル表現が得られるという特徴があります。

今回はその性質を利用して、言語間の転移学習の実験をしました。具体的には、ある自然言語処理のタスクの英語の訓練データセットを学習させ、他の言語のテスト用データセットで性能を評価しました。

## データセットの準備
今回実験に用いるデータセットは`TREC Question Classification Dataset`です。これは5452件の訓練セットと500件のテストセットからなり、英語の質問を6つの粗いカテゴリと42の細かいカテゴリで分類したものです。

配布されているデータセットは前処理としてcsvに変換しておきました。テストセットの先頭行は次のようになっています。

|label|text|
|:-:|:-:|
|NUM:dist|How far is it from Denver to Aspen ?|
|LOC:city|"What county is Modesto | California in ?"|
|HUM:desc|Who was Galileo ?|
|DESC:def|What is an atom ?|
|NUM:date|When did Hawaii become a state ?|
|NUM:dist|How tall is the Sears Building ?|
|HUM:gr|George Bush purchased a small interest in which baseball team ?|
|ENTY:plant|What is Australia 's national flower ?|
|DESC:reason|Why does the moon turn orange ?|

このテストデータセットはgoogle翻訳APIによって各言語に変換しました。変換後のデータセットの一例として日本語のものを示します。

|label|text|
|:-:|:-:|
|NUM:dist|デンバーからアスペンまでの距離は？|
|LOC:city|カリフォルニア州モデストとはどの郡ですか？|
|HUM:desc|ガリレオは誰ですか？|
|DESC:def|原子とは|
|NUM:date|ハワイはいつ国家になったのですか？|
|NUM:dist|シアーズビルの高さは？|
|HUM:gr|ジョージブッシュはどの野球チームにちょっとした興味を持っていますか？|
|ENTY:plant|オーストラリアの国花は何ですか？|
|DESC:reason|月がオレンジ色に変わるのはなぜですか。|

おかしな訳もあるものの、全体的に検証用のデータセットとしては十分なものが出来たと思います。

## 実験
tf-hubとtf.estimatorを使ってモデルを実装しました。モデルのコア部分のコードは次の通りです。入力文をtf-hubの多言語USEモジュールを使ってベクトル化したものを特徴量として、多層ニューラルネットワークで分類します。

```python
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
```

学習は英語の訓練データセットのみを使って行い、評価を多言語のテストセットで行いました。

## 結果
6つの大分類に分類した場合と、細かい42の小分類に分類した場合の両方を示します。

|言語|精度（大分類）|精度（小分類）|
|:-:|:-:|:-:|
|英語（訓練）|0.9473588|0.81969917|
|英語（テスト）|0.902|0.75|
|フランス語|0.842|0.7|
|日本語|0.882|0.714|
|スペイン語|0.904|0.732|
|韓国語|0.834|0.668|
|ドイツ語|0.876|0.742|
|中国語|0.894|0.732|
|イタリア語|0.852|0.694|
