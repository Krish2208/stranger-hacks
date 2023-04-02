from flask import Flask, render_template, request, url_for, redirect

import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from flask_gtts import gtts
import whisper
import pickle
app = Flask(__name__)
gtts(app)
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
whis_model = whisper.load_model('tiny')

# with open('blipmodel.pkl', 'rb') as f:
#     model = pickle.load(f)

# with open('blipprocessor.pkl', 'rb') as f:
#     processor = pickle.load(f)

# with open('whispermodel.pkl', 'rb') as f:
#     whis_model = pickle.load(f)

@app.route('/', methods=['GET'])
def index():
    return render_template('home.html')

@app.route('/blind', methods=['GET'])
def ren_blind():
    return render_template('blind.html')

@app.route('/deaf', methods=['GET'])
def ren_deaf():
    return render_template('deaf.html')

@app.route('/mute', methods=['GET'])
def ren_mute():
    return render_template('mute.html')


@app.route('/blind', methods=['POST'])
def blind():
    file = request.files['image']
    print(type(file))
    image = Image.open(request.files['image'])
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    text = processor.decode(out[0] , skip_special_tokens=True)
    return url_for(redirect('result_blind.html', text=text))

@app.route('/deaf', methods=['POST'])
def deaf():
    file = requests.files['audio']
    audio = file.read()
    text = whis_model.transcribe(audio)
    return url_for(redirect('result_deaf.html', text=text))

@app.route('/dumb', methods=['POST'])
def dumb():
    text = request.form['text']
    return url_for(redirect('result_mute.html', text=text))

if __name__ == '__main__':
    app.run(debug=True)