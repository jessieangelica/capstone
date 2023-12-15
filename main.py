import os
import pickle
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

app.config['MODEL'] = './models/intellicash-model.h5'
app.config['TOKENIZER'] = './models/tokenizer.pkl'

model = load_model(app.config['MODEL'], compile=False)
with open(app.config['TOKENIZER'], 'rb') as fileToken:
    publicToken = pickle.load(fileToken)

id2tag = {
    0: 'o',
    1: 'money',
    2: 'food',
    3: 'entertainment',
    4: 'electronics',
    5: 'clothing_accesories',
    6: 'transport_travel'
}

def make_prediction(preprocessed_sentence):
    sentence = [i.lower() for i in preprocessed_sentence.split()]
    sentence_id = publicToken.texts_to_sequences(sentence)
    len_orginal_sententce = len(sentence_id)
    padded_text = pad_sequences(sentence_id, maxlen=20, padding='post')
    prediction = model.predict(padded_text)
    prediction = np.argmax(prediction[0], axis=1)
    prediction = prediction[ : len_orginal_sententce]
    pred_tag_list = [id2tag[i] for i in prediction]
    return pred_tag_list

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        if(len(text) != 0):
            predict = make_prediction(text)
            return jsonify({
                'status': {
                    'code': 200,
                    'message': 'Success predicting',
                    'data': predict
                }
            }), 200
        else:
            return jsonify({
                'status': {
                    'code': 400,
                    'message': 'Body text is needed',
                }
            }), 400
    else:
        return jsonify({
            'status': {
                'code': 405,
                'message': 'Method not Allowed',
            }
        }), 405

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8000)))