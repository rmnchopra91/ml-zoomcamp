from flask import Flask, request, jsonify
import preprocess
import load_model as model

app = Flask("web_api")

@app.route('/home', methods=['GET'])
def test_server():
    print("this is home api..................")
    return preprocess.test()

@app.route('/')
def hello_world():
    print("this is get api..................")
    return 'Hello World'

@app.route('/predict', methods=['POST'])
def predict():
    print("this is predict API")
    data = request.get_json()
    is_data_valid, data = preprocess.remove_not_required_fields(data)
    if is_data_valid:
        y_pred = model.logisticregression().predict(data)
        return jsonify({"probability": float(y_pred)})
    return f"This is invalid request data: {data}"
    # valid_data, data = preprocess.remove_not_required_fields(data)


if __name__ == '__main__':
    app.run(debug=True, port=9696, host='0.0.0.0')