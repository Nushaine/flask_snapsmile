from flask import Flask, render_template, url_for, request, redirect, send_file, jsonify
import os
from model import testModel
import sys
from flask_cors import CORS, cross_origin


app = Flask(__name__)

# call model
directory = 'inputs'
imagesList = []

home_dir = os.getcwd()
UPLOAD_FOLDER = os.path.join(home_dir, "inputs")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
print(os.getcwd())

app.config["CORS_HEADERS"] = "Content-Type"


@app.route('/upload', methods=['GET', 'POST'])
@cross_origin()
def form():

    if request.method == "POST":

        if request.files:
            images = request.files.getlist("images")

            # Iterating through all uploaded images
            i = 0
            for image in images:
                image_name = "new_file_" + str(i) + ".jpg"
                image.save(os.path.join(
                    app.config['UPLOAD_FOLDER'], image_name))
                results = testModel()
                print(results)
                imagesList.append(results)
                print(imagesList)

                i += 1

            return send_file("yolov5/inference/output/new_file_0.jpg", mimetype="image/jpg")

    return render_template('test.html')


@app.route('/bbox', methods=['GET'])
def bbox_reciever():
    if request.method == "GET":
        with open("./yolov5/inference/data/data.txt", "r") as f:
            resultsBBOX = f.read()
        return jsonify(resultsBBOX)


@app.route('/upload/<int:id>', methods=['GET'])
def image_reciever(id):
    return send_file("yolov5/inference/output/new_file_" + str(id) + ".jpg", mimetype="image/jpg")


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
