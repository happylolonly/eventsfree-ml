import gensim

from gensim.corpora import Dictionary
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *


from html.parser import HTMLParser
stemmer = SnowballStemmer('russian')

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
        print(text);
        text = 'test';
    text = strip_tags(text);
    
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

tmp_fname = 'l23321';
# dict  ionary.save_as_text(tmp_fname)
dictionary = Dictionary.load_from_text(tmp_fname)

def big_func(text):
   


    # lda = lda_model
    # from gensim.test.utils import datapath

    # Save model to disk.
    # temp_file = datapath("./model")
    # lda.save('model')
    # Load a potentially pretrained model from disk.

    lda_model = gensim.models.LdaMulticore.load('model')

    unseen_document = 'How a Pentagon deal became an identity crisis for Google'
    bow_vector = dictionary.doc2bow(preprocess(text))

    l = []
    for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1*tup[1]):
        # print("Score: {}\t Topic: {}".format(score, lda_model.print_topic(index, 5)))
        d = {}
        d['score'] = score * 100;
        d['topic'] = lda_model.show_topic(index, 5);
        l.append(d)
    print('f', l)

    t = [];

    for i in l[0]['topic']:
        print(i);
        t.append({
            'name': i[0],
            'number': i[1] * 100,
        })

    print(t);
    return t;

# big_func('How a Pentagon deal became an identity crisis for Google')
import fastText

def predict(text):

    model2 = fastText.load_model('ft_quantized.3.model')

    t = preprocess(text)
    # print(t)

    print(" ".join(t))

    labels, probs = model2.predict([" ".join(t)], k=5)
    # labels, probs = model2.predict(test[FIELD].tolist())
    print(labels, probs)
    # labels = [ll.replace('__label__', '') for ll in labels]
    

    return {'labels': labels, 'probs': probs[0][0]}

    # print(classification_report(test['tag'].values, labels))


# predict('"Открытый Урок"! Фитнес-центр "Lifestyle Fitness & GYM" (г.Минск, пр-т.Машерова, 76а) представляет: серия бесплатных фитнес тренировок, направленных на популяризацию спорта в целом, спортивного образа жизни и даже спортивного мышления наших и не наших клиентов). 19.12.2017г, вторник, 19:30 Урок 1: "Тай Бо" Тренер: Кристина Санько-Вертоградова Вход свободный! Предварительная запись обязательно! +375 (44) 7-555-000, +375 (29) 500-10-55 info@fitness-club.by #ямамакрасивая #свояатмосфера #здесь')
