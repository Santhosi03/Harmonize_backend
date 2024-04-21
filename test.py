#Generally useful libraries
from dotenv import load_dotenv
import os
import requests
from github import Github

script_dir = os.path.dirname(os.path.abspath(__file__))

########Loading the SSL Certificate Chain and the Key File###############################

# from OpenSSL import SSL
import ssl

# Define SSL certificate and key file paths 
CERT_FILE = os.path.join(script_dir, "vmserver_fullchain.pem") 
KEY_FILE =  os.path.join(script_dir, "vmserver_key.pem")

# # Create SSL context v24.1... latest
# context = SSL.Context(SSL.TLSv1_2_METHOD) 
# context.use_certificate_chain_file(CERT_FILE)
# context.use_privatekey_file(KEY_FILE) 


#version 23 2023
# context = SSL.Context(SSL.PROTOCOL_TLSv1_2) 
# context.load_cert_chain(CERT_FILE, KEY_FILE) 


#ChatGPT SSL PYTHON PREINSTALLED
context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
context.load_cert_chain(certfile=CERT_FILE, keyfile=KEY_FILE)



####APP LOADING##################################################################
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
from gevent.pywsgi import WSGIServer
app = Flask(__name__)
CORS(app)
#################################################################################
#Here let us load the noise data into a dictionary
import pickle

noise_file = 'noises_29k.pkl' 
with open(noise_file, 'rb') as fp:
    noise_map = pickle.load(fp)
    print("done")
    print(len(noise_map))
###############MODEL AND TOKENIZER LOADING#######################################
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from keras.preprocessing.sequence import pad_sequences
import torch
import numpy as np


MODEL_NAME = "roberta_6_6_test=0.6_denoised.pth"
model_path = os.path.join(script_dir, MODEL_NAME)

# Load the pre-trained model

# model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model = RobertaForSequenceClassification.from_pretrained('roberta-base')

print("m loading")

# Load the saved model state
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
dev=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = model.to(dev)

print("model loaded")


# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Set the model to evaluation mode
model.eval()
print("set to eval mode")

#load the GitHub token and the HuggingFace Token
load_dotenv()

hf_token = os.environ.get('HF_TOKEN')
# print(hf_token)
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
headers = {"Authorization": "Bearer "+hf_token}
######################


###################################TOXICITY PREDICTION USING ROBERTA MODEL################################
def model_predict_dsh(sentence):
    new_sent = ""
    # print(sentence, " ::::::::::::Before")
    for word in sentence.split(" "): 
        word = word.lower()
        if word in noise_map:
            new_sent+= noise_map[word]
        else:
            new_sent+=word
        new_sent += " "
    sentence = new_sent.strip()
    # print(sentence, " :::::::::;After")
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



##################################REPHRASING TOXIC SENTENCES USING MISTRAL 7B######################################
def get_assistant_reply(output):
    # Split the conversation into lines

    lines = output[0]['generated_text'].split("\n")

    # Initialize a counter
    count = 0
    global step_2_size
    # Iterate through the lines from top to bottom
    for line in lines:
        # Check if the line contains the assistant's reply
        if line.startswith("    Assistant:"):
            # Increment the counter
            count += 1

            # Check if this is the fourth assistant's reply
            if count == step_2_size+1:
                # Extract and return the assistant's reply
                return line.strip().replace("Assistant:  ", "")

    # Return None if the fourth assistant's reply is not found
    return None


step_2_size=0
def model_suggest_san(toxic):

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()
    
#https://medium.com/@fractal.ai/create-conversational-agents-using-bloom-part-1-63a66e6321c0

    step_1 = "The following is a conversation with a detoxification expert. He had worked specifically in the \
        Software Engineering domain.Thus the expert will not consider 'kill the process' and similar SE specific\
        phrases as toxic. Expert helps the Human rephrase their toxic sentences into a non-toxic sentence so that\
        the target of the comment takes the constructive criticism and improves themselves. You are not supposed to \
        respond to the sentence but rephrase it.The expert is \
        conversational, optimistic, flexible, empathetic, creative and humanly in generating responses.\n"
    # step_1_5 = "You are \
    #     conversational, optimistic, flexible, empathetic, creative and humanly in generating responses.I am presenting a sample of a toxic comment found in a software engineering forum.\
    #             This is a toxic sentence,delimited by ---s: rewrite it in an untoxic form.\
    #             Please remember, these are comments extracted from Software Engineering Forums.\
    #             So they're from one user to another user. THEY ARE NOT DIRECTED TOWARDS YOU/GEMINI.\
    #             Remember that you should rewrite it in a software specific way as it is from a software forum:\
    #             Also remember, you MUST GIVE YOUR RESPONSE IN PLAIN TEXT, NO DELIMITATIONS."


    ##Step 2: n-shot prompting
    step_2 ="User: You and your code are shit.\n\
    Assistant: There is lot of improvement needed for your code\n\
    User: The fuck is your code.I can't understand shit.\n\
    Assistant: I am not able to understand your code. Can you explain it?\n\
    User: You niggas don't know how to print in java also.\n\
    Assistant: The way you are printing is incorrect. Java's syntax for print is different.\n"
    global step_2_size
    step_2_size=step_2.count('\n')//2
    step_3 = f"User: {toxic}\n\
    Assistant: "
    ##gemini version
    # step_3 = f"---{toxic}---"
    output = query({
        "inputs": step_1+step_2+step_3,
    })
    return get_assistant_reply(output)
    # prompt = "I am presenting a sample of a toxic comment found in a software engineering forum.\
    #                                        This is a toxic sentence,delimited by ---s: rewrite it in an untoxic form.\
    #                                        Please remember, these are comments extracted from Software Engineering Forums.\
    #                                       So they're from one user to another user. THEY ARE NOT DIRECTED TOWARDS YOU/GEMINI.\
    #                                       Remember that you should rewrite it in a software specific way as it is from a software forum:\
    #                                       Also remember, you MUST GIVE YOUR RESPONSE IN PLAIN TEXT, NO DELIMITATIONS.\
    #                                        ---" + toxic + "---"
    # print(prompt)
    # response = model_gem.generate_content(prompt, safety_settings=safety_settings)
    # print(response)
    # # print(response.prompt_feedback)
    # return response.text

#####FETCHING ISSUES AND COMMENTS FROM GITHUB AND CALCULATING THEIR TOXICITY SCORES##############
def model_repocheck(url):
    try:
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
                comments = issue.get_comments()
                for comment in comments:
                    if comment.body:  # Check if comment body is not None
                        o = model_predict_dsh(comment.body)
                        count+=1
                        toxic_count+=o[0]
                        toxic_prob+=o[1][0][1]
        print(toxic_count," toxic out of :",count)
        return toxic_prob/count
    except Exception as e:
        print("Error:", e)
        return None


############################################ADDING ROUTES TO THE BACKEND SERVER##############################
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        
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
        
        # Get suggestion
        toxic_comment = request.json
        if toxic_comment:
            result = model_suggest_san(toxic_comment)
            print("Suggetsions: ", result)
            return jsonify(result=result)
       

    return None

@app.route('/repocheck', methods=['GET', 'POST'])
def repocheck():
    print("out")
    if request.method == 'POST':
        
        # Make prediction for all comments in the repository
        
        url = request.json
        repository = url.split("github.com/")[-1]  # Extract everything after "github.com/"
        print(repository)
        
        print(request.json)
        prediction = model_repocheck(repository)
        result = prediction
        print("res: " , result)
        return jsonify(result=result)

    return None



#####################STARTING THE SERVER###################################################
if __name__ == '__main__':
    # m=model_predict_dsh("i don't need your opinion")
    # print(m[1][0][0])
    # k=model_repocheck("genyrosk/gym-chess")
    # print(k)
    print("s")
    app.run(host = "10.21.24.62",ssl_context=context)
    # http_server = WSGIServer(('10.23.105.29', 8080), app)
    print("h")
    # http_server.serve_forever()
