# Disaster Response Pipeline Project



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
