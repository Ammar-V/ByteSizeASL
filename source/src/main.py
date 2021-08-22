from flask import Flask, render_template, url_for, request, redirect
from Classifier import Classifier
import re
import base64
from words import WordSource

app = Flask(__name__)
clf = Classifier()
wordSource = WordSource()

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == "POST":
        try:
            base64_image = request.get_json(force=True)['imageBase64']
            image_data = re.sub('^data:image/.+;base64,', '', base64_image)
            image_bytes = base64.b64decode(image_data + "=")
            ans = clf.predict_sign(image_bytes)
        except Exception:
            return wordSource.get_random_word() 

        return ans

    return render_template('index.html', word=wordSource.get_random_word())

if __name__ == "__main__":
    app.run(debug = True, host = '0.0.0.0')