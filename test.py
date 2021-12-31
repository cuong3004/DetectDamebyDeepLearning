import torch 
import tensorflow as tf
import pandas as pd
from albumentations.pytorch import ToTensorV2
from PIL import Image
import albumentations as A
import numpy as np 
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools


model_file_name = "model.tflite"
file_extension = model_file_name.split('.')[-1]

df_test = pd.read_csv("test.csv")

labels_dir = {
    "Acne" : 0,
    "Eczema" : 1,
    "Nail_Fungus" : 2,
    "Psoriasis" : 3,
    "Normal": 4
}


transform_test_pt = A.Compose([
    A.Resize(224,224),
    A.Normalize(mean=127.5, std=127.5, max_pixel_value=1.0),
    ToTensorV2(),
])

transform_test_tflite = A.Compose([
    A.Resize(224,224),
    A.Normalize(mean=127.5, std=127.5, max_pixel_value=1.0)
])

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')




if file_extension == "pt":
    model = torch.jit.load(model_file_name)

    all_preds = []

    for name_file in tqdm(df_test["Path Image"]):
        img = np.array(Image.open(name_file).convert("RGB"))

        img = transform_test_pt(image=img)["image"]
        img_input = torch.unsqueeze(img, 0)
        with torch.no_grad():
            pred = model(img_input)
        # print(pred)
        # assert False
        pred = pred.argmax(dim=1)
        all_preds.append(pred)

    all_preds = torch.cat(all_preds,dim=0)

    cm = confusion_matrix(df_test["Index Value"].values, all_preds)

    plt.figure(figsize=(10,10))
    plot_confusion_matrix(cm, list(labels_dir), normalize=True)
    plt.savefig("test_pt.png")
    plt.clf()


elif file_extension == "tflite":
    interpreter = tf.lite.Interpreter(model_path=model_file_name)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("input_detial", input_details)
    print("output_details", output_details)

    all_preds = []

    for name_file in tqdm(df_test["Path Image"]):

        img = np.array(Image.open(name_file).convert("RGB"))

        img = transform_test_tflite(image=img)["image"]
        img_input = np.expand_dims(img, 0)

        interpreter.set_tensor(input_details[0]['index'], img_input)
        interpreter.invoke()

        pred = interpreter.get_tensor(output_details[0]['index'])

        pred = pred.argmax(axis=1)
        all_preds.append(pred)
        
        # assert False

    all_preds = np.concatenate(all_preds, axis=0)

    cm = confusion_matrix(df_test["Index Value"].values, all_preds)

    plt.figure(figsize=(10,10))
    plot_confusion_matrix(cm, list(labels_dir), normalize=True)
    plt.savefig("test_tflite.png")
    plt.clf()
else:
    raise "model error"
    











