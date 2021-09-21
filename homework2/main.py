# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os
import nltk
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity

sw = stopwords.words('russian')
sw.append('это')
sw.append('то')
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def collecting():
    curr_dir = os.getcwd()
    friends_dir = os.path.join(curr_dir, 'friends-data')
    list_of_episodes = []
    for root, dirs, files in os.walk(friends_dir):
        for name in files:
            list_of_episodes.append(os.path.join(root, name))
    return list_of_episodes
def preproccess(filepath, request):
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
    if request == 0:
        collection_of_texts.append(current_text)
        print('done')
    if request == 1:
        return current_text


        # Press the green button in the gutter to run the script.

def indexing(corpus):
    global vectorizer, X
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    return X.toarray()

def get_request():
    value = input("Please enter a request for search:\n")
    return value
def continuing():
    value = input("Do you want to continue (yes/no):\n")
    if value == 'yes':
        showing(searching(vectors, index_request(get_request())))
    if value == 'no':
        return 0
def index_request(request_string):
    with open('request.txt', 'w', encoding='UTF-8') as f:
        f.write(request_string)
    request_list = []
    request_list.append(preproccess('request.txt', 1))
    vector_request = vectorizer.transform(request_list).toarray()
    return vector_request

def searching(vectors, vector_request):
    s1 = cosine_similarity(vectors, vector_request, 'cosine')
    s2 = [g[0] for g in s1]
    return s2
def showing(similarity_list):
    indexes = list(reversed(list(np.argsort(similarity_list))))
    for index in indexes:
        print(list_of_episodes[index].split('\\')[-1])
    continuing()


if __name__ == '__main__':
    collection_of_texts = []
    print('stge1')
    list_of_episodes = collecting()
    for episode in list_of_episodes:
        preproccess(episode, 0)
    print('stge2')
    vectors = indexing(collection_of_texts)
    print('stge3')
    value = get_request()
    showing(searching(vectors, index_request(value)))
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
