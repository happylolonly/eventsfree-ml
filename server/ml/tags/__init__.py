import fastText

model = fastText.load_model('ml/tags/model/ft_quantized.model')

def predict(text):
    labels, probs = model.predict([text], k=5)

    print(labels, probs)

    tags = [];

    for i, label in enumerate(labels[0]):
        tags.append({
            'tag': label.replace('__label__', ''),
            'probability': round(probs[0][i], 3)
        })

    return tags
