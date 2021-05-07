from flask import Flask, jsonify, request
from flask_restful import Api, Resource
import bcrypt

app = Flask(__name__)
api = Api(app)

# DB Connection Code
from pymongo import MongoClient
client = MongoClient("mongodb://127.0.0.1:27017")
db = client.TestDataBase
UserSchema = db["User"]


def verifyPW(username, password):
    hashed_pw = UserSchema.find({
        "username": username
    })[0]["password"]

    if bcrypt.hashpw(password.encode('utf-8'), hashed_pw) == hashed_pw:
        return True
    else:
        return False
    

def tokenCounts(username):
    tokens = UserSchema.find({
        "username": username
    })[0]["tokens"]
    return tokens


class Register(Resource):
    def post(self):
        data = request.get_json(force=True)
        username = data["username"]
        password = data["password"]
        sentence = ""
        print(password)
        hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        UserSchema.insert({
            "username": username,
            "password": hashed_pw,
            "sentence": sentence,
            "tokens": 3
        })
        
        return jsonify({
            "Status": 200,
            "message": "Successfully Registered"
        })


class StoreSentence(Resource):
    def post(self):
        data = request.get_json(force=True)
        username = data["username"]
        password = data["password"]
        sentence = data["sentence"]

        correct_pw = verifyPW(username, password)
        if not correct_pw:
            return jsonify({

                "status": 302,
                "message": "Invalid User and Password"
            })

        enough_tokens = tokenCounts(username)
        if enough_tokens <= 0:
            return jsonify({

                "status": 302,
                "message": "Not Enough Tokens"
            })

        UserSchema.update({
            "username": username
        }, {
            "$set": {
                "sentence": sentence,
                "tokens": enough_tokens - 1
            } 
        })
        


        return jsonify({
            "Status": 200,
            "message": "Sentence Saved Successfully"
        })

api.add_resource(Register, "/register-user")
api.add_resource(StoreSentence, "/register-sentence")
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=90, debug=True)



        # status = validateUserData(data, "add")
        # print(status)
        # if (status != 200):
        #     return jsonify({
        #         "error": "Parameters are missing",
        #         "success": 0
        #     })
        

# End of DB Connection Code