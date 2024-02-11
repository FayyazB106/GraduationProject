# Graduation Project - Dr AI's Assistant

## Introduction
This is a web app developed for my graduation project in early 2023. It uses a deep learning model to predict the presence of cancer by analyzing chest CT scans. "Dr AI's Assistant" is the working title.

## Description
The app is mostly Python. The deep learning model is trained using TensorFlow in Google Colab. The model is saved then imported into `app.py` which is a Flask app. The front-end is rendered using HTML and CSS. A [dataset](https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images) of 1000 images was used to train the model.

## Usage
To train the model like I did, open the `AI_model.ipynb` notebook in Google Colab and import the dataset given above. Run all blocks of code and download the .h5 file generated in the end. Place the .h5 file in the same directory as `app.py`.

Create a virtual environment using the `requirements.txt` file.
To launch the app, run `app.py` using the terminal.
```
python app.py
```
The app will run in localhost and the address will be displayed in the terminal.

## License
License: [MIT](https://opensource.org/licenses/MIT)
