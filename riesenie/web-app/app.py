import os
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

try:
    import shutil
    shutil.rmtree('uploads')
    os.mkdir('uploads')
except:
    pass

app = Flask(__name__)

MODEL_PATH = 'models/model_EfficientNetB1_acc96_valacc92_testacc95.h5'

model = load_model(MODEL_PATH)

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(240, 240))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    return preds

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('upload.html')
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        preds = model_predict(file_path, model)

        pred_class = preds.argmax(axis=-1)
        result = int(pred_class[0])

        labels = ['Muchotrávka oranžová', 'Hríb dubový', 'Kuriatko jedlé', 'Hnojník obyčajný', 
        'Lievik trúbkovitý', 'Chriapač kučeravý', 'Bedľa vysoká', 'Hliva ustricová', 
        'Muchotrávka červená', 'Muchotrávka tigrovaná', 'Muchotrávka končistá', 'Ušiak obyčajný', 
        'Strapcovka zväzkovitá', 'Rýdzik kravský', 'Prilbička reďkovková', 'Čírovka zemná']

        edibility = ""
        if result <= 7:
            edibility = "Jedlá"
        else:
            edibility = "Jedovatá"

        label = labels[result]

        if max(preds[0])*100 < 50:
            label = "nerozpoznany"
            edibility = "nerozpoznany"
        
        return render_template('upload.html', label = label, edibility = edibility, percentage = "{:.2f}".format(max(preds[0])*100))
    return None


if __name__ == '__main__':
    app.run(debug=True)

