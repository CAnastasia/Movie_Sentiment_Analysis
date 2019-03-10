from joblib import load

def predict_rf(sentence):
    vectorizer = load('models/random_forest_vectorizer.joblib')
    random_forest = load('models/random_forest_model.joblib')
    pred = random_forest.predict(vectorizer.transform([sentence]))[0]
    return str(pred)


if __name__ == "__main__":
    pass