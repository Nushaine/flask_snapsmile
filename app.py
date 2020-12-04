from flask import Flask, render_template, url_for, request, redirect, send_file, send_from_directory, jsonify, make_response
import os
from model import testModel
from PIL import Image
import sys
import shutil
import base64
from flask_cors import CORS, cross_origin


app = Flask(__name__)

directory = 'inputs'
home_dir = "D:/work/snapsmile/computer visoin/flask/"
UPLOAD_FOLDER = os.path.join(home_dir, "inputs")
UPLOAD_FOLDER = str(os.getcwd()) + "/inputs"

home_dir = os.getcwd()
UPLOAD_FOLDER = os.path.join(home_dir, "inputs")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

print(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config["CORS_HEADERS"] = "Content-Type"


@app.route('/upload', methods=['GET', 'POST'])
def form():
    if request.method == "POST":
        print('Sending Post Request')
        print(request.files)
        if request.files:
            print("REQUEST HAS FILES", request.files)
            images = request.files.getlist("images")
            print("IMAGES", images)

            # Iterating through all uploaded images
            for i, image in enumerate(images):
                if len(images) != 0:
                    print("FORLOOP STARTED")
                    image_name = "new_file_" + str(i) + ".jpg"
                    image_path = os.path.join(
                        app.config['UPLOAD_FOLDER'], image_name)
                    # saving all uploaded images to 'inputs' folder
                    image.save(image_path)
                    print("SAVED IMAGE TO PATH ", image_path)
                else:
                    return "Request had no iamges"

            results = testModel()

            outputs_path = home_dir + '/yolov5/inference/output/'
            print(outputs_path)
            full_path = os.path.join(outputs_path, image_name)
            print("FULL PATh", full_path)
            return send_file(full_path, mimetype="image/jpg")

        print('Request doesnt have files')
    return render_template('test.html')


@app.route('/classes', methods=['GET', 'POST'])
def detections():
    if request.method == "POST":
        print(request.form)
        print("REQUEst path", request.path)
        class_name = request.form['class']
        print("CLASS NAME ", class_name)
        full_path = os.path.join('yolov5/yolov5/inference/output/', class_name)
        print("FULL PATH ", full_path)
        image_path = os.path.join(full_path, 'detections.jpg')
        print("IMAGE PATH ", image_path)

        # with open(image_path, "rb") as image_file:
        #encoded_string = base64.b64encode(image_file.read())
        #print("ENCODEDSTRING", encoded_string)
        #response = make_response(send_file(image_path))

        return send_file(image_path, mimetype="image/jpg")
    return render_template('testv2.html')


@app.route('/bbox', methods=['GET'])
def bbox_reciever():
    if request.method == "GET":
        with open("./yolov5/inference/data/data.txt", "r") as f:
            resultsBBOX = f.read()
        return jsonify(resultsBBOX)


@app.route("/true_false", methods=["GET"])
def true_false():
    if request.method == "GET":
        with open("./yolov5/inference/data/true_false.txt", "r") as f:
            resultsTrue = f.read()
        resultsTrue = resultsTrue.replace("[", "")
        resultsTrue = resultsTrue.replace("]", "")
        resultsTrue = resultsTrue.replace(",", "")
        resultsTrue = resultsTrue.split(" ")
        return_dict = {}
        return_dict['Cavity Square'] = resultsTrue[0]
        return_dict['Gingivitis'] = resultsTrue[1]
        return_dict['Plaque'] = resultsTrue[2]
        return_dict['Stains'] = resultsTrue[3]
        return(jsonify(return_dict))


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)), debug=True)
