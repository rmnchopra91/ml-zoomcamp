from flask import Flask, request, jsonify
import pickle

app = Flask('credict-card')


#load the model
def load_model(model):
    with open(model, 'rb') as file:
        model = pickle.load(file)
        return model

dv = load_model('./dv.bin')
model = load_model('./model1.bin')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data from request
        data = request.get_json()
        print("this is predict api------------------")
        # Perform Prediction
        X = dv.transform([data])
        y_pred = model.predict_proba(X)[0,1]
        get_card = y_pred >= 0.5

        result = {
            'get_card_probability': float(y_pred),
            'get_card': bool(get_card)
        }

        # Return the prediction as json
        return jsonify(result)
    except Exception as e:
        print("this is exception part in predict API")
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
    

