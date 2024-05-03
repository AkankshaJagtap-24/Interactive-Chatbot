from flask import Flask, render_template, request, jsonify
import nltk
import json
import random
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from keras.models import load_model 
from flask_cors import CORS

model = load_model('chatbot_model_v2.h5')
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words_v2.pkl','rb'))
classes = pickle.load(open('classes_v2.pkl','rb'))
 
app = Flask(__name__)
CORS(app)

@app.get("/")
def index_get():
    return render_template("C:/Users/NITIN/Documents/Interactive-Chatbot-1/templates/base.html")


@app.route("/run")
def run_scripts():
    file =open('C:/Users/NITIN/Desktop/AI interactive agent/backend train_chatbot_v2.py','r').read()
    return exec(file)

@app.route("/predict",methods=['POST'])
def predict():
    text= request.get_json().get("message")
    response=get_bot_response(text)
    message={"answer": response}
    return jsonify(message)

@app.route("/get")
def get_bot_response(show_details=True):
    file =open(r'C:/Users/NITIN/Documents/Interactive-Chatbot-1/backend train_chatbot_v2.py','r').read()
    xy = exec(file)
    userText = request.args.get('msg')
    return str(xy.chatbot.get_response(userText))

    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(userText)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]

    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    p = np.array(bag)
    
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    
    ints = return_list
    
    tag = ints[0]['intent']
    list_of_intents = intents.json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return str(result.lower())


if __name__ == "__main__":
    app.run(debug=True)