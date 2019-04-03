import random
import time


from flask import Flask, request, jsonify, redirect, url_for

import sys

sys.path.append('gc/')

from run import generate_image


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

    """
    di = {
        'concept':request.form['name'],
        'src':"tmp/{}.jpg".format(request.form['identifier']),
        'caption':'summary of settings'
        }
    time.sleep(5)
    """
    di = generate_image(request.form)

    return jsonify(di)