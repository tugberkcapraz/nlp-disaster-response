import sys
import pandas as pd
# For Database connection
from sqlalchemy import create_engine
# For NLP related tasks
import re
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
nltk.download('omw-1.4')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
# For data modelling and evaluating pipeline
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, f1_score, make_scorer
from sklearn.model_selection import train_test_split
# Saving the model
import pickle




def load_data(database_filepath, table_name):
    """
    A function to load data from database and split it into predictors and target. It takes path to database and related table name from that data base
    as params.Returns predictors, target and target labels.
    :param database_filepath: Path to database
    :param table_name: Table name that contains the data
    :return: Predictors, target, target labels
    """

    # Read the data from sql
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table(table_name, engine)
    df.drop('child_alone', axis=1, inplace=True) # No positive observations here.
    # split features and target
    X = df.iloc[:,1].values
    y = df.iloc[:,4:].values
    # target labels.
    category_names = df.iloc[:,4:].columns

    return X, y, category_names


def tokenize(text):
    """

    :param text: input text to be tokenized
    :return: tokenized text
    """
    # normalize case and remove punctuation
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text.lower())

    # tokenize text and save them
    tokens = word_tokenize(text)

    # initiate lemmatizer and define stopwords
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words("english")

    # remove spaces around word (stripping).And then, Lemmatize the words that are not in stop_words list.
    clean_tokens = [lemmatizer.lemmatize(word.strip()) for word in tokens if word not in stop_words]

    return clean_tokens

def build_model():
    """
    Model builder pipeline that
    1) Vectorizes the text using the custom tokenizer function above.
    2) Passes the vectorized text to TfidfTransformer.
    3) Passes the transformed text into CatBoostClassifier
    4) Grid Search Cross validator splits data into 3 folds and aims to optimize the fit by learning rate adjustments"
    :return: Returns the best fitting model given the grid of hyper parameters that were provided.
    """
    # Make f1 scorer
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('classifier',MultiOutputClassifier(CatBoostClassifier(iterations=10,
                                                               verbose=True)))
    ])
    # hyper-parameter grid
    parameters = {#'classifier__estimator__depth': [4,5],
                  'classifier__estimator__learning_rate':[0.1, 0.3]}

    # create model
    model = GridSearchCV(estimator=pipeline,
                     param_grid=parameters,
                     verbose=3,
                     cv=3)

    return model


def evaluate_model(model, X_test, y_test, category_names):
    """
    A function to evaluate performance of ML model.
    :param model: Model object
    :param X_test: Test data to make predictions on.
    :param y_test: Test labels to compare predictions
    :param category_names: Test labels in human-understandable format.
    :return: Classification report for the ML model.
    """

    y_pred = model.predict(X_test)
    classification_report(y_test, y_pred, target_names=category_names )


def save_model(model, model_filepath):
    """
    Model saver function.
    :param model: ML model object to save
    :param model_filepath: filepath to save the ML model object
    :return:
    """
    with open(model_filepath, "wb") as file:
        pickle.dump(model, file)



def main():
    if len(sys.argv) == 4:
        database_filepath, table_name, model_filepath = sys.argv[1:]
        print('Loading data...\n   from DATABASE: {} of tablename: {}'.format(database_filepath, table_name))
        X, y, category_names = load_data(database_filepath, table_name)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True)

        print('Building model...')
        model = build_model()
        
        print('Training model...')

        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument. The TableName from the given database should '\
              'be passed as second argument. And the filepath of the pickle file to '\
              'save the model is the third argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db PostETL ClassifierName.pkl')


if __name__ == '__main__':
    main()