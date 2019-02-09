import fastText
import dropbox_helper

try:
    dropbox_helper.load('./ml/tags/model/tags_model_new', '/tags_model_new')
    model = fastText.load_model('ml/tags/model/tags_model_new')
    print('loaded last tags model')
except ValueError:
    model = fastText.load_model('ml/tags/model/tags_model')
    print('loaded default tags model')


def predict(text):
    
    labels, probs = model.predict([text], k=5)

    print(labels, probs)

    tags = [];

    for i, label in enumerate(labels[0]):
        tags.append({
            'label': label.replace('__label__', ''),
            'probability': round(probs[0][i], 3)
        })

    return tags


# fast code for result
# def build_model() {
    
# }
