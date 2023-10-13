import pickle

# model_path = 'model1.bin'

def load(model_path):
    with open(model_path, 'rb') as model_instance:
        model = pickle.load(model_instance)
        return model

dv = load("./dv.bin")
model = load('./model1.bin')

client = {"reports": 0, "share": 0.001694, "expenditure": 0.12, "owner": "yes"}

X = dv.transform([client])
y_pred = model.predict_proba(X)[0, 1]

print(f"Here is thr result: {y_pred}")
# print(y_pred)