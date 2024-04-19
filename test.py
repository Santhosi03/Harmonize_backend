from transformers import BertTokenizer
from keras.preprocessing.sequence import pad_sequences

from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
from gevent.pywsgi import WSGIServer

import google.generativeai as genai
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "model2.pth")

from github import Github

from dotenv import load_dotenv, dotenv_values 
load_dotenv()
github_token = os.getenv('GITHUB_TOKEN')
print(github_token)

#checking git push


###################################
safety_settings = [
    {
        "category": "HARM_CATEGORY_DANGEROUS",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]
genai.configure(api_key="AIzaSyDEWOQzsQZSILCax2fnrGbkmMKC2xBHOsE")
model_gem = genai.GenerativeModel('gemini-pro')
############################

import torch
import numpy as np


####APP LOADING####
app = Flask(__name__)
CORS(app)
############

# ###MODEL AND TOKENIZER LOADING###
from transformers import BertForSequenceClassification

# Load the pre-trained model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
print("m loading")
# Load the saved model state
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
print("model loaded")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# Set the model to evaluation mode
model.eval()
print("set to eval mode")
######################



def model_predict_dsh(sentence):
    tokenized_sentence = tokenizer.tokenize(sentence)


    MAX_LEN = 128

    # Use the BERT Tokenizer to convert the tokens to their index numbers in the BERT vocabulary
    input_ids = tokenizer.convert_tokens_to_ids(tokenized_sentence)

    # Pad the input tokens
    padded_input = pad_sequences([input_ids], maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

    # Create attention mask
    attention_mask = [float(i > 0) for i in padded_input[0]]



    input_tensor = torch.tensor(padded_input)
    attention_mask_tensor = torch.tensor([attention_mask])


    # Make predictions
    with torch.no_grad():
        outputs = model(input_tensor, attention_mask=attention_mask_tensor)

    # Get the predicted class probabilities
    predicted_probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    # Get the predicted class index
    predicted_class = torch.argmax(predicted_probs, dim=1).item()
    # print(predicted_class, predicted_probs.numpy())
    return (predicted_class,predicted_probs.numpy())


def model_suggest_san(toxic):
    prompt = "I am presenting a sample of a toxic comment found in a software engineering forum.\
                                           This is a toxic sentence,delimited by ---s: rewrite it in an untoxic form.\
                                           Please remember, these are comments extracted from Software Engineering Forums.\
                                          So they're from one user to another user. THEY ARE NOT DIRECTED TOWARDS YOU/GEMINI.\
                                          Remember that you should rewrite it in a software specific way as it is from a software forum:\
                                          Also remember, you MUST GIVE YOUR RESPONSE IN PLAIN TEXT, NO DELIMITATIONS.\
                                           ---" + toxic + "---"
    print(prompt)
    response = model_gem.generate_content(prompt, safety_settings=safety_settings)
    print(response)
    # print(response.prompt_feedback)
    return response.text


def model_repocheck(url):
    try:
        score_list=[]
        # Initialize PyGithub with an anonymous GitHub API access
        github_token = os.environ.get('GITHUB_TOKEN')

# Initialize the GitHub client with the token
        g = Github(github_token)

        # Get the repository
        repo = g.get_repo(url)
        count=0
        toxic_count=0
        toxic_prob=0
        # Get all issues from the repository
        issues = repo.get_issues(state='all')

        # Iterate through issues and extract comments
        for issue in issues:
            if issue.body:  # Check if issue body is not None
                p = model_predict_dsh(issue.body)
                count+=1
                toxic_count+=p[0]
                toxic_prob+=p[1][0][1]
                score_list.append(p[1][0][1])
                comments = issue.get_comments()
                for comment in comments:
                    if comment.body:  # Check if comment body is not None
                        o = model_predict_dsh(comment.body)
                        count+=1
                        toxic_count+=o[0]
                        toxic_prob+=o[1][0][1]
                        score_list.append(o[1][0][1])
        print(toxic_count," ",count)
        print(score_list)
        return toxic_prob/count
    except Exception as e:
        print("Error:", e)
        return None



@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        # img = base64_to_pil(request.json)
        
        # Make prediction
        print("hi")
        prediction = model_predict_dsh(request.json)
        pred = prediction[1]
        result = prediction[0]
        pred_probability = "{:.3f}".format(np.amax(pred)) 
        
        return jsonify(result=result, probability=pred_probability)

    return None


@app.route('/suggest', methods=['GET', 'POST'])
def suggest():
    # print("out")
    if request.method == 'POST':
        # Get the image from post request
        # img = base64_to_pil(request.json)
        
        # Make prediction
        print(request.json)
        prediction = model_suggest_san(request.json)
        result = prediction
        print("res: " , result)
        return jsonify(result=result)

    return None

@app.route('/repocheck', methods=['GET', 'POST'])
def repocheck():
    print("out")
    if request.method == 'POST':
        # Get the image from post request
        # img = base64_to_pil(request.json)
        
        # Make prediction
        
        url = request.json
        repository = url.split("github.com/")[-1]  # Extract everything after "github.com/"
        print(repository)
        
        print(request.json)
        prediction = model_repocheck(repository)
        result = prediction
        print("res: " , result)
        return jsonify(result=result)

    return None


if __name__ == '__main__':
    # m=model_predict_dsh("i don't need your opinion")
    # print(m[1][0][0])
    # k=model_repocheck("tensorflow/tensorflow")
    # print(k)
    print("s")
    http_server = WSGIServer(('127.0.0.1', 5000), app)
    print("h")
    http_server.serve_forever()
