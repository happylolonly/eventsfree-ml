import gensim
from gensim.corpora import Dictionary


dictionary = Dictionary.load_from_text('ml/lda/model/dict')
lda_model = gensim.models.LdaMulticore.load('ml/lda/model/model')

def predict(text):   
    bow_vector = dictionary.doc2bow(text)

    l = []
    for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1*tup[1]):
        # print("Score: {}\t Topic: {}".format(score, lda_model.print_topic(index, 5)))
        d = {}
        d['score'] = score;
        d['topic'] = lda_model.show_topic(index, 5);
        l.append(d)

    data = []

    for i in l[0]['topic']:
        print(i)
        data.append({
            'label': i[0],
            'probability': round(i[1] * 100 / 100, 3) # какой то костыль с делением, по другому хз, ошибка float32
        })

    return data
