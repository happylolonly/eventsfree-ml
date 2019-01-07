from flask import Flask, abort, request, jsonify

import os
import ml


app = Flask(__name__)

@app.route('/lda', methods=['POST'])
def lda():
    if not request.json:
        abort(400)

    data = request.get_json()
    prediction = ml.predictLDA(data['text'])

    print(prediction)

    resp = jsonify(prediction)
    return resp


@app.route('/tags', methods=['POST'])
def tags():
    if not request.json:
        abort(400)

    data = request.get_json()
    prediction = ml.predictTags(data['text'])

    print(prediction)

    resp = jsonify(prediction)
    return resp


@app.route('/train', methods=['GET'])
def train():
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor

    with open('../notebooks/tags/tags.ipynb') as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=600, kernel_name='python3', log_level=0)

    print('train')
    ep.preprocess(nb, {'metadata': {'path': '../notebooks/tags'}})
    return ''





if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
