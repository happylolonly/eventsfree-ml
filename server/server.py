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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
