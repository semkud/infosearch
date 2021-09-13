# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os
import nltk
from nltk.corpus import stopwords
sw = stopwords.words('russian')
sw.append('это')
sw.append('то')
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

def collecting():
    curr_dir = os.getcwd()
    friends_dir = os.path.join(curr_dir, 'friends-data')
    list_of_episodes = []
    for root, dirs, files in os.walk(friends_dir):
        for name in files:
            list_of_episodes.append(os.path.join(root, name))
    return list_of_episodes
def preproccess(filepath):
    global collection_of_texts
    inp = filepath
    outp = "./"
    output_filename = os.path.join(os.path.abspath(outp), 'temp.txt')
    mystem_path = os.path.join('/Users/SKudr/PycharmProjects/infosearch1/', 'mystem')
    os.system(f"""{mystem_path} "{inp}" "{output_filename}" -lnd """)
    current_text = ''
    alphabet='абвгдеёжзийклмнопрстуфхцчшщъыьэюя-'
    with open(output_filename, 'r', encoding= 'UTF-8') as f:
        text = f.read().splitlines()
        for line in text:
            word = ''
            for symbol in line:
                if symbol in alphabet:
                    word+=symbol
            if word not in sw:
                current_text+=word+" "
    collection_of_texts.append(current_text)


        # Press the green button in the gutter to run the script.

def index_reverse(corpus):
    vectorizer = CountVectorizer(analyzer='word')
    X = vectorizer.fit_transform(corpus)
    return pd.DataFrame(X.toarray(), columns = vectorizer.get_feature_names())

def some_stats(df):
    freq_dict = {}
    everywhere_dict = {}
    for word in list(df.columns):
        freq_dict[word] = df[word].sum()
        everywhere_dict[word] = df[word].min()
    sort_down = dict(sorted(freq_dict.items(), key=lambda item: item[1], reverse=True))
    sort_up = dict(sorted(freq_dict.items(), key=lambda item: item[1], reverse=False))
    print('Топ-20 самых частотных слов:')
    for k in list(sort_down)[:20]:
        print(k, ' --', sort_down[k], 'штук')
    print('Топ-20 самых редких слов:')
    for k in list(sort_up)[:20]:
        print(k, ' --', sort_up[k], 'штук')
    print('Список слов, которые есть во всех документах:')
    for key in everywhere_dict:
        if everywhere_dict[key] != 0:
            print(key)
    heroes_popularity = {}
    heroes_popularity['моника'] = freq_dict['моника'] + freq_dict['мон']
    heroes_popularity['рэйчел'] = freq_dict['рэйчел'] + freq_dict['рэйч'] + freq_dict['рэй']
    heroes_popularity['чендлер'] = freq_dict['чендлер'] + freq_dict['чен']
    heroes_popularity['фиби'] = freq_dict['фиби'] + freq_dict['фибс']
    heroes_popularity['росс'] = freq_dict['росс']
    heroes_popularity['джоуи'] = freq_dict['джоуи'] + freq_dict['джо'] + freq_dict['джой']

    print('Самые популярные герои:')
    sort_hero = dict(sorted(heroes_popularity.items(), key=lambda item: item[1], reverse=True))
    for key in sort_hero:
        print(key, '--', sort_hero[key])

if __name__ == '__main__':
    collection_of_texts = []
    list_of_episodes = collecting()
    for episode in list_of_episodes:
        preproccess(episode)
    df = index_reverse(collection_of_texts)
    some_stats(df)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
