# -*- coding: utf-8 -*-

import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *

from html.parser import HTMLParser
stemmer = SnowballStemmer('russian')

import ml.lda
import ml.tags


def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ''.join(self.fed)

def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()

def preprocess(text):
    if (type(text) == float):
        text = 'test'
        
    text = strip_tags(text);
    
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result


def predictLDA(text):
    text = preprocess(text)
    return lda.predict(text)


def predictTags(text):
    text = " ".join(preprocess(text))
    return tags.predict(text)


# predictLDA('How a Pentagon deal became an identity crisis for Google')
# predictTags('"Открытый Урок"! Фитнес-центр "Lifestyle Fitness & GYM" (г.Минск, пр-т.Машерова, 76а) представляет: серия бесплатных фитнес тренировок, направленных на популяризацию спорта в целом, спортивного образа жизни и даже спортивного мышления наших и не наших клиентов). 19.12.2017г, вторник, 19:30 Урок 1: "Тай Бо" Тренер: Кристина Санько-Вертоградова Вход свободный! Предварительная запись обязательно! +375 (44) 7-555-000, +375 (29) 500-10-55 info@fitness-club.by #ямамакрасивая #свояатмосфера #здесь')
