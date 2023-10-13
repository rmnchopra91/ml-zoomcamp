from flask import Flask, jsonify
import pickle

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        return 'responce done'
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run()