import fastText


def predict(text):

    try:
        model = fastText.load_model('ml/tags/model/tags_model_new') # fix
    except ValueError:
        model = fastText.load_model('ml/tags/model/tags_model') # fix
    
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
