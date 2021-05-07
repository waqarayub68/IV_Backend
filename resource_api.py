from flask import Flask, jsonify, request
from flask_restful import Api, Resource
import os

app = Flask(__name__)
api = Api(app)

# DB Connection Code
from pymongo import MongoClient
client = MongoClient("mongodb://127.0.0.1:27017")
db = client.aNewDB
UserNum = db["UserNum"]
UserNum.insert({
    "noofUsers": 0
})
# End of DB Connection Code




def validateUserData(data, functionName):
    if (functionName == 'add'):
        print(data)
        if "a" not in data or "b" not in data:
            return 301
        else:
            return 200



class Add(Resource):
    def post(self):
        data = request.get_json(force=True)
        status = validateUserData(data, "add")
        print(status)
        if (status != 200):
            return jsonify({
                "error": "Parameters are missing",
                "success": 0
            })
        
        return jsonify({
            "SUM": data["a"] + data["b"],
            "success": 1
        })

    def get(self):
        pass

    def put(self):
        pass

    def delete(self):
        pass

class Substract(Resource):
    pass

class Multiply(Resource):
    pass

class Divide(Resource):
    pass

api.add_resource(Add, "/add")

@app.route('/')
def hello():
    return "Hello Waqar"


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=90, debug=True)