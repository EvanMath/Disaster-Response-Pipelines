import sys
import pandas as pd
import pickle
import re

from sqlalchemy import create_engine

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def load_data(database_filepath):
    """
    This function loads the database, reads the table and returns feature and target values.

    INPUT:
    database_filepath: File to database

    OUTPUT
    X: Pandas dataframe with feature 'message'
    Y: Pandas dataframe with target values
    category_names
    """

    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('CleanMessages', engine)

    X = df['message']
    Y = df.iloc[:, 4:]

    return X, Y, Y.columns


def tokenize(text):
    """
    Helper function to tokenize the messages
    INPUT:
    text: Messages

    OUTPUT:
    Tokens of word
    """

    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()

    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    tokens = word_tokenize(text)

    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens


def build_model():
    """
    Initiate a Multioutput Classifier with Random Forest Classifier,
    create a pipeline for feature engineering and using Grid Search to find the best params
    for model.

    OUTPUT:
    Model to be fitted
    """

    clf = MultiOutputClassifier(RandomForestClassifier())
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer(smooth_idf=False)),
        ('clf', clf)
    ])

    parameters = {
        'clf__estimator__min_samples_leaf': [2, 5],
        'clf__estimator__max_depth': [None, 10, 50],
        'clf__estimator__n_estimators': [10, 20, 40],
        'clf__estimator__min_samples_split': [2, 3, 4],
    }

    model_pipeline = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1, verbose=2)

    return model_pipeline


def evaluate_model(model, X, Y, category_names):
    """
    INPUT:
    model: Pipeline model to be trained.
    X: Pandas dataframe to be used for train test split
    Y: Pandas dataframe to be used for train test split
    category_names: Category names

    OUTPUT:
    Fitted model
    """

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.3, random_state=0)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    numOfColumns = len(category_names)

    for idx in range(numOfColumns):
        print(y_test.columns[idx])
        print(classification_report(y_test.iloc[:, 0], y_pred[:, 0]))
        print('--------------')

    return model


def save_model(model, model_filepath):
    """
    INPUT:
    model: Fitted model with best parameters
    model_filepath: File path to be saved
    """

    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
