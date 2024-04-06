#import mediapipe as mp
from flask import Flask, request, render_template
app = Flask(__name__)



@app.route('/test', methods = ['GET'])
def drawskeleton():
    r = {"data": "Hello Universe", "headers": {"Access-Control-Allow-Origin": "*"}}
    return r

if __name__ == "__main__":
    app.run(debug=True) 