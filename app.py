from flask import Flask, render_template, request, redirect, flash, url_for
from tensorflow import keras
from keras.applications import efficientnet
import os
from werkzeug.utils import secure_filename
import numpy as np

# Initialization and configuration
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/upload'
app.config['SECRET_KEY'] = 'hfisuyhfi'

# Limit the type of extensions allowed for upload
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load the default homepage
@app.route('/')
def home():
    dir = 'static/upload'
    # Empty the static/upload directory each time new session is opened
    for file in os.scandir(dir):
        os.remove(file.path)
    return render_template('index.html')

# Process image and display result  
@app.route('/', methods=['POST'])
def process():
    # Ensure no empty upload
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    # Upload image
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename)) 

        # Import AI model and initialize labels
        ai_model = keras.models.load_model("ai_model.h5")
        classnames = ['adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib', 'large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa', 'normal', 'squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa']
        
        # Preprocess image
        tf_img = keras.utils.load_img('static/upload/'+ filename, target_size=(500, 500))
        tf_img = keras.utils.img_to_array(tf_img)
        tf_img = np.expand_dims(tf_img, axis=0)
        tf_img = efficientnet.preprocess_input(tf_img)
        
        # Prediction and retrieval of result
        pred = ai_model.predict(tf_img)
        prob = pred[0]

        # Display image and result. Resutlt inconclusive if max probability is less than 70%
        if np.max(prob)*100 >= 70:
            if classnames[np.argmax(prob)] == 'adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib':
                flash(f"Adenocarcinoma detected with {np.max(prob)*100:.2f}% confidence.")
            if classnames[np.argmax(prob)] == 'large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa':
                flash(f"Large-cell carcinoma detected with {np.max(prob)*100:.2f}% confidence.")
            if classnames[np.argmax(prob)] == 'squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa':
                flash(f"Squamous-cell carcinoma detected with {np.max(prob)*100:.2f}% confidence.")
            if classnames[np.argmax(prob)] == 'normal':
                flash(f"No cancer detected with {np.max(prob)*100:.2f}% confidence.")
            return render_template('index.html', filename=filename)
        else:
            flash('Inconclusive result')
            return render_template('index.html', filename=filename)
    else:
        flash('Allowed image types are: .png, .jpg, .jpeg')
        return redirect(request.url)
    
# Run app
if __name__ == "__main__":
    app.run()