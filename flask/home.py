from flask import Flask
from flask import jsonify
from flask import render_template
import number_captcha

app = Flask(__name__)
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/captcha_number')
def function_1():
    json = number_captcha.captcha_number()
    return json

@app.route('/recognize_number/<dir>')
def function_2(dir):
    json = number_captcha.recognize(dir)
    return json

if __name__ == '__main__':
    app.run()