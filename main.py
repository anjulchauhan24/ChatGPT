from flask import Flask, render_template, jsonify, request

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api", methods=["GET", "POST"])
def qa():
    if request.method=="POST":
        print(request.json)
        question = request.json("question")
        data={"result":f"answer of{question}"}
        return jsonify(data)
    data={"result":"Hey!  Its great to hear from you. Hows everything going? Let me know whats on your mind—Im here to help with anything you need!"}
        
    return jsonify(data)


app.run(debug=True)