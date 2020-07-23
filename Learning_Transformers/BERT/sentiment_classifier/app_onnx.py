import config
# import dataset
import torch

from flask import Flask, jsonify
from flask import request
from model import BERTBasedUncased
import onnxruntime as ort
import numpy as np
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

app = Flask(__name__)

MODEL = ort.InferenceSession("sent_bert_model.onnx")

# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def to_numpy(tensor):
    if tensor.requires_grad:
        return tensor.detach().cpu().numpy()
    else:
        return tensor.cpu().numpy()

def sentence_prediction(sentence):
    tokenizer = config.TOKENIZER
    max_len = config.MAX_LEN
    review = str(sentence)
    
    inputs = tokenizer.encode_plus(
            review, 
            None,
            add_special_tokens=True,
            max_length = max_len,
            truncation = True
        )

    ids = inputs["input_ids"]
    mask = inputs["attention_mask"]
    token_type_ids = inputs["token_type_ids"]

    # BERT We need to padd on right side.
    padding_length = max_len - len(ids)
    ids = ids + ([0] * padding_length)
    mask = mask + ([0] * padding_length)
    token_type_ids = token_type_ids + ([0] * padding_length)

    ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    mask = torch.tensor(mask, dtype=torch.long).unsqueeze(0)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).unsqueeze(0)
    
    onnx_inputs = {"ids": to_numpy(ids), "mask": to_numpy(mask), "token_type_ids": to_numpy(token_type_ids)}
    output = MODEL.run(None, onnx_inputs)
    return sigmoid(output[0][0])


@app.route("/predict")
def predict():
    sentence = request.args.get("sentence")
    print(sentence)
    positive_prediction = sentence_prediction(sentence)
    negative_prediction = 1 - positive_prediction
    response = {}
    response["response"] = {
        "positive_prediction" : str(positive_prediction),
        "negative_prediction" : str(negative_prediction),
        "sentence" : str(sentence)
    }

    return jsonify(response)


if __name__ == "__main__":
    # output = sentence_prediction("awesome video")
    app.run(debug=True, port=8001)
