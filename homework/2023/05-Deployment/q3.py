import pickle

def load_model(model_path):
    with open(model_path, 'rb') as model:
        return pickle.load(model)

print(f"Model load start......................")
dv = load_model('./dv.bin')
model = load_model('./model1.bin')
print(f"model loaded sucessfully.........................")

input_data = {"job": "retired", "duration": 445, "poutcome": "success"}
X = dv.transform(input_data)
y_pred = model.predict_proba(X)[0,1]

print(f"==============prediction======={y_pred}")