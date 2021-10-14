from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as f
import gensim
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import json
from scipy import sparse
import os
from sklearn.preprocessing import normalize

def mean_pooling(model_output):
    return model_output[0][:,0]

def building_corpus():
    answers = []
    questions = []
    with open('questions_about_love.jsonl', 'r', encoding='UTF-8') as f:
        jsontext = list(f)
    for a in range(10000):
        try:
            answers.append(json.loads(jsontext[a])['answers'][0]['text'])
            questions.append(json.loads(jsontext[a])['question'] +' '+ json.loads(jsontext[a])['comment'])
        except:
            pass
    return answers, questions

def preprocess_corpus(corpus):
    sw = stopwords.words('russian')
    sw.append('это')
    sw.append('то')

    longstring = ''
    for text in corpus:
        longstring+=text + ' этомойразделитель '
    with open('temp_string.txt', 'w', encoding='UTF-8') as f:
        f.write(longstring)
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

def bert_vectorizing(corpus):
    tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
    model = AutoModel.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
    for a in range(0, len(corpus)-30, 25):
        print(a)
        encoded_input = tokenizer(corpus[a:a+25], padding=True, truncation=True, max_length=24, return_tensors='pt')
        with torch.no_grad():
            model_output = model(**encoded_input)
        if a == 0:
            sentence_embeddings = mean_pooling(model_output)
        else:
            sentence_embeddings = torch.cat((sentence_embeddings, mean_pooling(model_output)),dim =0)
    del tokenizer, model
    return sentence_embeddings

def fasttext_vectorizing(corpus):
    model = gensim.models.KeyedVectors.load('araneum_none_fasttextcbow_300_5_2018.model')
    corpus_vec = []
    for text in corpus:
        cur_vecs = []
        tokens = word_tokenize(text)
        for token in tokens:
            try:
                cur_vecs.append(model[token])
            except:
                pass
        if len(cur_vecs) != 1:
            corpus_vec.append(np.mean(np.array(cur_vecs), axis = 0))
        else:
            corpus_vec.append(np.array([0]*300))
    del model
    return torch.Tensor(corpus_vec)

def bm25_vectorizing(corpus):
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

def bm25_scoring(corpus, query):
    result = 0
    query_vec = count_vectorizer.transform(query)
    scores = np.dot(corpus, query_vec.T).toarray()
    for a in range(len(scores)):
        sorted_scores_indx = np.argsort(scores[a], axis=0)[::-1]
        if a in sorted_scores_indx[0:5]:
            result+=1
    return result/len(scores)

def tf_idf_vectorizing(corpus):
    global tfvectorizer, X
    tfvectorizer = TfidfVectorizer()
    X = tfvectorizer.fit_transform(corpus)
    return X

def tf_idf_scoring(answers, questions):
    result = 0
    scores = np.dot(answers, questions.T).toarray()
    for a in range(len(scores)):
        sorted_scores_indx = np.argsort(scores[a], axis=0)[::-1]
        if a in sorted_scores_indx[0:5]:
            result += 1
    return result / len(scores)

def count_vectorizing(corpus, querry):
    global cvvectorizer, X1
    cvvectorizer = CountVectorizer()
    X1 = cvvectorizer.fit_transform(corpus)
    q1 = cvvectorizer.transform(querry)
    return X1, q1


def scoring(answers, questions):
    result = 0
    scores = np.array(torch.matmul(answers, questions.T))
    for a in range(len(scores)):
        sorted_scores_indx = np.argsort(scores[a], axis=0)[::-1]
        if a in sorted_scores_indx[0:5]:
            result+=1
    return result/len(scores)


if __name__ == '__main__':
    scores = {}
    corpus_ans, corpus_que = building_corpus()
    #counting bert-score:
    ans_vec = f.normalize(bert_vectorizing(corpus_ans), p=2, dim=1)
    que_vec = f.normalize(bert_vectorizing(corpus_que), p=2, dim=1)
    scores['bert'] = scoring(ans_vec, que_vec)
    #counting fasttext:
    ans_vec = f.normalize(fasttext_vectorizing(corpus_ans), p=2, dim=1)
    que_vec = f.normalize(fasttext_vectorizing(corpus_que), p=2, dim=1)
    scores['fasttext'] = scoring(ans_vec, que_vec)
    #counting bm-25:
    ans_proc = preprocess_corpus(corpus_ans)
    que_proc = preprocess_corpus(corpus_que)
    ans_vec = bm25_vectorizing(ans_proc)
    scores['bm25'] = bm25_scoring(ans_vec, que_proc)
    #counting tf-idf:
    ans_vec = tf_idf_vectorizing(ans_proc)
    scores['tfidf'] = tf_idf_scoring(ans_vec, tfvectorizer.transform(que_proc))
    #counting count-vec:
    ans_vec, que_vec = count_vectorizing(ans_proc, que_proc)
    scores['countvec'] = tf_idf_scoring(normalize(ans_vec, norm='l1', axis=1), normalize(que_vec, norm='l1', axis=1))
    with open('score_results.txt', 'w', encoding='UTF-8') as f:
        for key in scores.keys():
            f.write(str(key)+' : '+str(scores[key])+'\n')
