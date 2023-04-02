from flask import Flask, render_template, request, url_for, redirect

from gtts import gTTS
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
    image = Image.open(file.stream)
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    text = processor.decode(out[0] , skip_special_tokens=True)
    audio = gTTS(text)
    audio.save('static/audio.mp3')
    return render_template('result_blind.html')

@app.route('/deaf', methods=['POST'])
def deaf():
    file = requests.files['audio']
    audio = file.read()
    text = whis_model.transcribe(audio)
    return url_for(redirect('result_deaf.html', text=text))

@app.route('/mute', methods=['GET', 'POST'])
def dumb():
    print(request.form)
    text = request.form['text']
    audio = gTTS(text)
    audio.save('static/audio.mp3')
    return render_template('result_mute.html')

if __name__ == '__main__':
    app.run(debug=True)