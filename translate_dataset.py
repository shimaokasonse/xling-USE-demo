from google.cloud import translate
import pandas as pd

def transform(dataset_df, target_language):
    """
    dataset_df: pandas dataframe which has columns [label, text].
    target_language: target language code to translate
    """
    translate_client = translate.Client()
    translated_texts = []
    for text in dataset_df.text.values:
        result = translate_client.translate(text, target_language=target_language)
        translated_texts.append(result['translatedText'])
        print(result)
    
    df = pd.DataFrame({'text':translated_texts})
    dataset_df.update(df)

    return dataset_df
    
languages = ["zh-CN","ja","fr","ko","de","it","es"]
for lang in languages:
    df = pd.read_csv("test.csv")
    new_df = transform(df,lang)
    new_df.to_csv("test_{}.csv".format(lang),index=False)