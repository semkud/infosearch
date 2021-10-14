from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as f
import numpy as np
import json
import gensim
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

def mean_pooling(model_output):
    return model_output[0][:,0]

def building_corpus():
    corpus = []
    with open('questions_about_love.jsonl', 'r', encoding='UTF-8') as f:
        jsontext = list(f)
    for a in range(50000): #На 50.000 я не тестировал, и так скорее всего он упадет по памяти. Надо подавать текст на анализ батчами, как в соседнем скрипте. Но ответ на поставленную задачу  - этот код.
        try:
            #Я беру первый по очереди ответ, а не от самого умного участника, концептуально для работы программы разницы нет
            corpus.append(json.loads(jsontext[a])['answers'][0]['text'])
        except:
            pass
    return corpus

def bert_vectorizing(corpus):
    tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
    model = AutoModel.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
    encoded_input = tokenizer(corpus, padding=True, truncation=True, max_length=24, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output)
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
    return torch.Tensor(corpus_vec)

def get_request():
    value = input("Please enter a request for search:\n")
    return value

def continuing():
    value = input("Do you want to continue (yes/no):\n")
    if value == 'yes':
        #searching(corpus_vec_norm, f.normalize(bert_vectorizing([get_request()]), p = 2, dim = 1))
        searching(corpus_vec_norm, f.normalize(fasttext_vectorizing([get_request()]), p=2, dim=1))
    if value == 'no':
        return 0

def searching(corpus_vec_norm, query):
    scores = np.array(torch.matmul(corpus_vec_norm, query[0]))
    sorted_scores_indx = np.argsort(scores, axis=0)[::-1]
    result = np.array(corpus)[sorted_scores_indx.ravel()]
    print(result[:10])
    continuing()


if __name__ == '__main__':
    #Стояли задачи сделать поисковик для фасттекста и берта, соответственно написаны оба варианта
    #Сейчас откоменчен фасттекст и законмечен берт
    corpus = building_corpus()
    #corpus_vec = bert_vectorizing(corpus)
    corpus_vec = fasttext_vectorizing(corpus)
    corpus_vec_norm = f.normalize(corpus_vec, p = 2, dim = 1)
    #searching(corpus_vec_norm, f.normalize(bert_vectorizing([get_request()]), p = 2, dim = 0))
    searching(corpus_vec_norm, f.normalize(fasttext_vectorizing([get_request()]), p = 2, dim = 0))
