from flask import Flask,render_template,request
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from tensorflow.keras.models import load_model
import pickle
import numpy as np

upload_folder = "static"

def predict(filepath):
    label_mapping = {"NORMAL":0,"COVID":1}

    label_inv_mapping = dict([(v,k) for k,v in label_mapping.items()])

    with open("models/head_model.pkl","rb") as f:
        head_model = pickle.load(f)

    base_model = load_model("models/resnet_model.h5")

    img = load_img(filepath,target_size=(224,224))
    img = img_to_array(img)
    img = np.expand_dims(img,axis = 0)
    img = preprocess_input(img)

    features = base_model.predict(img).reshape(1,-1)

    label = head_model.predict(features)

    max_probs = np.max(head_model.predict_proba(features).reshape(-1))
    pred_class = label_inv_mapping[label[0]]
    accuracy = float(str(max_probs*100)[:7])
    return pred_class,accuracy


app = Flask(__name__)

@app.route("/", methods=["GET","POST"])
def upload_predict():
    if request.method == "POST":
        image_file = request.files["image"]
        if image_file:
            image_location = os.path.join(upload_folder,image_file.filename)
            image_file.save(image_location)
            pred_class,accuracy = predict(image_location)
            print(pred_class,accuracy)
            return render_template("index.html",image_loc = image_file.filename,pred_class=pred_class,accuracy=str(accuracy))
    return render_template("index.html",image_loc = None,pred_class=None,accuracy=None)

if __name__ == "__main__":
    app.run(debug=True)



