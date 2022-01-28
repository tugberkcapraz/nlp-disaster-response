import sys
import pandas as pd
# For Database connection
from sqlalchemy import create_engine
# For NLP related tasks
import re
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer




def load_data(database_filepath, table_name):
    """
    A function to load data from database and split it into predictors and target. It takes path to database and related table name from that data base
    as params.Returns predictors, target and target labels.
    :param database_filepath: Path to database
    :param table_name: Tablename that contains the data
    :return: Predictors, target, target labels
    """

    # Read the data from sql
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table(table_name, engine)

    # Split X and y
    X = df["message"]
    y = df.drop(['id', 'message', 'original', 'genre'], axis =1)
    y_labels = y.columns

    return X, y, y_labels


def tokenize(text):
    """
    A helper function to tokenize and lemmatize the text.
    The function also removes the urls and hashes from the text before tokenization.
    :param text: Text to be tokenized & lemmatized
    :return: tokenized text
    """

    # There are plenty of text that are containing url.
    # replace each url in text string with empty string. Putting there a placeholder is risky. We are going to
    # convert everything into numbers in the following steps. I don't want my classifier to make spurious associations
    url_placeholder = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_placeholder, text)
    text = [text.replace(url, ' ') for url in detected_urls]
    #for url in detected_urls:
    #    text = text.replace(url,' ')

    # there are #s in the text.
    text = text.replace('#', ' ')

    # Now tokenize text
    tokens = word_tokenize(text)

    # Call lemmatizer object
    lemmatizer = WordNetLemmatizer()

    # turn every token into lower case and strip off white spaces before lemmatization.
    clean_tokens = []
    for raw_token in tokens:
        processed_token = lemmatizer.lemmatize(raw_token.lower().strip())
        clean_tokens.append(processed_token)
    # NOTE: list comprehension here was making it really hard to read

    return clean_tokens


def build_model():



def evaluate_model(model, X_test, Y_test, category_names):
    pass


def save_model(model, model_filepath):
    pass


def main():
    if len(sys.argv) == 4:
        database_filepath, table_name = sys.argv[1:]
        print('Loading data...\n   from DATABASE: {} of tablename: {}'.format(database_filepath, table_name))
        X, Y, category_names = load_data(database_filepath, table_name)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(table_name))
        save_model(model, table_name)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()