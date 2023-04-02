from gtts import gTTS
import os

def speak(text):
    tts = gTTS(text=text, lang='en')
    filename = 'voice.mp3'
    tts.save(filename)
    os.system('mpg321 ' + filename)