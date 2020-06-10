# Disaster-Response-Pipelines

This project contains three components.

**1.** **ETL Pipeline**

  `process_data.py` contains the steps of an ETL Pipeline that:
  - Loads the `messages` and `categories` datasets
  - Merges the two datasets
  - Cleans the data
  - Stores it in a SQLite database
  
 **2.** **ML Pipeline**
 
  `train_classifier.py` is a python script contains a machine learning pipeline that:
  - Loads data from the SQLite database
  - Splits the dataset into training and test sets
  - Builds a text processing and machine learning pipeline
  - Trains and tunes a model using GridSearchCV
  - Outputs results on the test set
  - Exports the final model as a pickle file
  
**3.** **Flask Web App**
 
  Reads the database and the saved machine learning model to create visualizations and predict messages in website.
  
## Instructions
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
  
    - To run ML pipeline that trains classifier and saves python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
    
2. Run the following command in the app's directory to run your web app. python run.py

Go to http://127.0.0.1:3001/

# Prerequisites
You will need to hae install the following libraries:
* Pandas
* Sqlalchemy
* Plotly
* Flask
* Sklearn
* Nltk
* Re (Regular Expressions)

You will also need to download for the nltk package these:
- nltk.download('punkt')
- nltk.download('stopwords')
- nltk.download('wordnet')
