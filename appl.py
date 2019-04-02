import random

from flask import Flask, request, render_template, jsonify, redirect, url_for, send_from_directory

import time
import werkzeug


#with open('spam_model.pkl', 'rb') as f:
#    model = pickle.load(f)
app = Flask(__name__, static_url_path="")

@app.route('/')
def index():
    return redirect(url_for('static',filename='index.html'))

@app.route('/graphic-clouds/')
def gc():
    return redirect(url_for('static',filename='graphic-clouds/index.html'))

@app.route('/clouds-submit', methods=['POST'])
def create():
    """Return a random prediction."""
    elems = [
        'identifier','name','engine','t_im_range','transform','colors',
        'im_width','im_height','font1','font2','font3','font4','inverted'
        ]
    text = ""
    for each in elems:
        text += each + ' ' + str(request.form[each]) + "\n"

    di = {
        'concept':request.form['name'],
        'src':"tmp/{}.jpg".format(request.form['identifier']),
        'caption':'summary of settings'
        }
    time.sleep(5)
    return jsonify(di)