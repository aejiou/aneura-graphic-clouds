from flask import Flask, request, jsonify, redirect, url_for

import sys
import os

sys.path.append('gc/')

from run import generate_image


app = Flask(__name__, static_url_path="")
application = app

@app.route('/')
def index():
    return redirect(url_for('static',filename='index.html'))

@app.route('/graphic-clouds/')
def gc():
    return redirect(url_for('static',filename='graphic-clouds/index.html'))

@app.route('/clouds-submit', methods=['POST'])
def create():

    di = generate_image(request.form)

    os.remove('static/tmp/'+request.form['identifier']+'.log')

    return jsonify(di)

@app.route('/clouds-status', methods=['GET'])
def update():
    with open("static/tmp/"+request.args.get('id')+".log") as f:
        for line in f:
            return line

if __name__ == "__main__":
    app.run()


