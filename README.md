# Data-Engnering-Project

Project summary
The goal of this research is to develop a model for categorizing messages transmitted during emergencies. There are 36 pre-established categories, some of which include Aid Related, Medical Help, Search and Rescue, and so on. We can enable the proper disaster aid organization to get these messages by categorizing them. To make the work easier, this project will entail creating a simple ETL and machine learning pipeline. Given that a message may fall under more than one category, this classification task also uses several labels.


Lastly, this project includes a web application that allows you to input a message and receive results for classification.

## File Description
~~~~~~~
disaster_response_pipeline
app:
templates
go.html
master.html
run.py

data:
disaster_message.csv
disaster_categories.csv
DisasterResponse.db
process_data.py

models:
classifier.pkl
train_classifier.py

Preparation:
categories.csv
ETL Pipeline Preparation.ipynb
ETL_Preparation.db
messages.csv
ML Pipeline Preparation.ipynb
EADME
README
~~~~~~~

## installation
Must runing with Python 3 with libraries of numpy, pandas, sqlalchemy, re, NLTK, pickle, Sklearn, plotly and flask libraries.

## File Descriptions
App folder including the templates folder and "run.py" for the web application
Data folder containing "DisasterResponse.db", "disaster_categories.csv", "disaster_messages.csv" and "process_data.py" for data cleaning and transfering.
Models folder including "classifier.pkl" and "train_classifier.py" for the Machine Learning model.
README file
Preparation folder containing 5 different files, which were used for the project building. (Please note: this folder is not necessary for this project to run.)
## Instructions
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Licensing, Authors, Acknowledgements
Many thanks to Figure-8 for making this available to Udacity for training purposes. Special thanks to udacity for the training. Feel free to utilize the contents of this while citing me, udacity, and/or figure-8 accordingly.

### NOTICE: Preparation folder is not necessary for this project to run.
