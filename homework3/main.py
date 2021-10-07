import os
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import json
from scipy import sparse
sw = stopwords.words('russian')
sw.append('это')
sw.append('то')

def preprocess(string):
    global sw
    with open('temp_string.txt', 'w', encoding='UTF-8') as f:
        f.write(string)
    inp = 'temp_string.txt'
    outp = "./"
    output_filename = os.path.join(os.path.abspath(outp), 'temp_preproc_string.txt')
    mystem_path = os.path.join(os.getcwd(), 'mystem')
    os.system(f"""{mystem_path} "{inp}" "{output_filename}" -lnd """)
    current_text = ''
    alphabet='абвгдеёжзийклмнопрстуфхцчшщъыьэюя-'
    corpus_proc = []
    with open(output_filename, 'r', encoding= 'UTF-8') as f:
        text = f.read().splitlines()
        for line in text:
            if line != 'этомойразделитель?':
                word = ''
                for symbol in line:
                    if symbol in alphabet:
                        word+=symbol
                if word not in sw:
                    current_text+=word+" "
            else:
                corpus_proc.append(current_text)
                current_text = ''
    return corpus_proc

def building_corpus():
    #corpus_proc = []
    corpus_virgin = []
    with open('questions_about_love.jsonl', 'r', encoding='UTF-8') as f:
        jsontext = list(f)[:50000]
    for a in range(0, 50000):
        print(a)
        try:
            #corpus_proc.append(preprocess(json.loads(jsontext[a])['answers'][0]['text']))
            corpus_virgin.append(json.loads(jsontext[a])['answers'][0]['text'])
        except:
            pass
    return np.array(corpus_virgin)

def indexing(corpus):
    global x_count_vec, count_vectorizer
    count_vectorizer = CountVectorizer()
    tf_vectorizer = TfidfVectorizer(use_idf=False, norm='l2')
    tfidf_vectorizer = TfidfVectorizer(use_idf=True, norm='l2')

    x_count_vec = count_vectorizer.fit_transform(corpus)
    x_tf_vec = tf_vectorizer.fit_transform(corpus)
    x_tfidf_vec = tfidf_vectorizer.fit_transform(corpus)
    idf = tfidf_vectorizer.idf_

    values = []
    rows = []
    cols = []
    k = 2
    b = 0.75
    len_d = x_count_vec.sum(axis=1)
    avdl = len_d.mean()

    for i, j in zip(*x_tf_vec.nonzero()):
        values.append(float(idf[j] * x_tf_vec[i, j] * (k + 1) / (x_tf_vec[i, j] + (k * (1 - b + b * len_d[i] / avdl)))))
        rows.append(i)
        cols.append(j)

    sparce_matrix = sparse.csr_matrix((values, (rows, cols)))
    return sparce_matrix

def get_request():
    value = input("Please enter a request for search:\n")
    return value

def continuing():
    value = input("Do you want to continue (yes/no):\n")
    if value == 'yes':
        searching(matrix, preprocess(get_request()+' этомойразделитель ')[0])
    if value == 'no':
        return 0

def searching(sparce_matrix, query):
    query_count_vec = count_vectorizer.transform([query])
    scores = np.dot(sparce_matrix, query_count_vec.T).toarray()
    sorted_scores_indx = np.argsort(scores, axis=0)[::-1]
    result = corpus_virgin[sorted_scores_indx.ravel()]
    print(result[:10])
    continuing()


#так как майстем очень долго обрабатывает все подокументно,
# я решил слить все документы в один и обработать разом
def prc():
    longstring = ''
    for text in corpus_virgin:
        longstring+=text + ' этомойразделитель '
    corpus_proc = preprocess(longstring)
    return corpus_proc

if __name__ == '__main__':
    corpus_virgin = building_corpus()
    corpus_proc = prc()
    matrix = indexing(corpus_proc)
    print(type(matrix))
    searching(matrix, preprocess(get_request()+' этомойразделитель ')[0])