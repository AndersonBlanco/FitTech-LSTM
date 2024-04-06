#import mediapipe as mp
from flask import Flask, request, render_template, jsonify
app = Flask(__name__)



@app.route('/test', methods = ['GET'])
def drawskeleton():
    render_template("build")


app.run(port=3000)