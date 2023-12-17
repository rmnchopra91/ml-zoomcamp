import pickle

def logisticregression(model_path):
    with open(model_path, 'rb') as model:
        return pickle.load(model)