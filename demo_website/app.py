from flask import Flask
from flask_restful import Api, Resource
from flask_cors import CORS

from models.random_forest import predict_rf
from helpers.random_sentence import get_random_sentence

app = Flask(__name__)
CORS(app)
api = Api(app)


class RandomForest(Resource):
    def get(self, sentence):
        return predict_rf(sentence)

class RandomSentence(Resource):
    def get(self):
        return get_random_sentence()

# Correspondance URL : modèle

api.add_resource(RandomForest, '/random_forest/<string:sentence>')
api.add_resource(RandomSentence, '/random_sentence/')
