from flask import Flask,render_template,request
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from tensorflow.keras.models import load_model
import pickle
import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import transforms
from PIL import Image
import cv2
import torchvision.models as models


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

#Neural Network Prediction - EfficientNetB0
# def predict(filepath):
#     PATH = "/home/jash/Desktop/JashWork/COVIDCT_Flask/models/efficientnet.pt"
#     label_mapping = {"NORMAL":0,"COVID":1}

#     label_inv_mapping = dict([(v,k) for k,v in label_mapping.items()])

#     class CovModel(nn.Module):
#         def __init__(self):
#             super(CovModel,self).__init__()

#         def get_model(self):
#             resnet_model = models.efficientnet_b0(pretrained=True)
#             resnet_model.classifier = nn.Sequential(nn.Linear(1280,256),
#                                             nn.ReLU(inplace = True),
#                                             nn.Dropout(0.75),
#                                             nn.Linear(256,2))
            
#             return resnet_model
        
#     model = CovModel().get_model()
#     model.load_state_dict(torch.load(PATH, map_location=torch.device("cpu")))

#     normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])
#     common_transforms = transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         normalize
#     ])

#     # img = cv2.imread(filepath)
#     # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#     # img = img.astype(np.float32)/255.0
#     # img = np.transpose(img, (2, 0, 1))
#     img = Image.open(filepath).convert("RGB")
#     x = common_transforms(img)
#     x = x.unsqueeze(0)
#     with torch.no_grad():
#         model.eval()
#         pred = model(x)
#         num = torch.argmax(pred,dim = 1).item()
#         pred_prob = pred[0,num].item()
    

#     return label_inv_mapping[num],pred_prob*100

    
    


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



