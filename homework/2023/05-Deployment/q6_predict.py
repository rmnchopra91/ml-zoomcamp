from flask import Flask, jsonify, request
import pickle

def load_model(model_path):
    with open(model_path, 'rb') as model:
        return pickle.load(model)

app = Flask('prediction')

dv = load_model('./dv.bin')
model = load_model('./model2.bin')

@app.route('/')
def hello_world():
    return 'Hello World'

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.get_json()
    print("this is predict api------------------")
    X = dv.transform([input_data])
    print(f"data converted into vector----------{X}--------")
    y_pred = model.predict_proba(X)[0,1]
    print(f"prediction is ready----------{y_pred}--------")
    return jsonify({'probability' : float(y_pred)})

if __name__ == '__main__':
    app.run(debug=True, port=9696, host='0.0.0.0')