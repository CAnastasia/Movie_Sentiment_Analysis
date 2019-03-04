#!flask/bin/python
from flask import Flask, jsonify
from joblib import load

app = Flask(__name__)

tasks = [
    {
        'id': 1,
        'title': u'Buy groceries',
        'description': u'Milk, Cheese, Pizza, Fruit, Tylenol', 
        'done': False
    },
    {
        'id': 2,
        'title': u'Learn Python',
        'description': u'Need to find a good Python tutorial on the web', 
        'done': False
    }
]

@app.route('/api/tasks', methods=['GET'])
def get_tasks():
    return jsonify({'tasks': tasks})

@app.route('/api/random_forest/predict/<string:sentence>', methods=['GET'])
def predict(sentence):
    random_forest = load('random_forest/random_forest.joblib')
    vectorizer = load('random_forest/vectorizer.joblib')
    pred = random_forest.predict(vectorizer.transform([sentence]))
    return str(pred)

if __name__ == '__main__':
    app.run(debug=True)
