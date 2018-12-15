import os
import ml

print(ml.big_func)



from flask import Flask, abort, request, jsonify
import json


    


app = Flask(__name__)

# @app.route("/")
# def hello():
#     return "Hello World!"

@app.route('/', methods=['GET', 'POST'])
def parse_request():
    # if not request.json:
    #     abort(400)
    o = request.get_json()
    print(o)
    # print(type(o))
    # d = json.loads(o)
    l = ml.big_func(o['text']);
    print(l)

    d = l;

    message = {
        'status': 200,
        'message': 'OK',
        'scores': d
    }
    resp = jsonify(message)
    resp.status_code = 200
    print(resp)
    return resp

    # return json.dumps(l)
    # data = request.args  # data is empty
    # print(data)
    # return ml.big_func('How a Pentagon deal became an identity crisis for Google')
    # need posted data here

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
