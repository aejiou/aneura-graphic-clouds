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
    #elems = [
    #    'identifier','name','engine','t_im_range','transform','colors',
    #    'im_width','im_height', ,'inverted'
    #    ]
    #text = ""
    #for each in elems:
    #    text += each + ' ' + str(request.form[each]) + "\n"

    #return '123'

    di = generate_image(request.form)

    os.remove('static/tmp/'+request.form['identifier']+'.log')
    f=open("requests.csv", "a+")
    f.write(di['log'])
    f.close()

    return jsonify(di)

@app.route('/clouds-status', methods=['GET'])
def update():
    with open("static/tmp/"+request.args.get('id')+".log") as f:
        for line in f:
            return line

if __name__ == "__main__":
    app.run()


