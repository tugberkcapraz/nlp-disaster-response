import json
import os
import sys
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import sklearn.externals
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('PostEtl', engine)
print(df.head())
# load model
model = joblib.load("../models/CatClassifier.pkl")
print("model_loaded")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for first Graph
    genre_df = df.groupby('genre')['message'].count().reset_index()
    genre_names = genre_df["genre"]
    genre_counts = genre_df["message"]

    # create the graph
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts,
                    width=.5,
                    textfont = {'family' : 'Arial'},
                    marker=dict(color='silver')
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"

                },
                'padding': 150

            },

        }
    ]
    ## second graph
    # extract data needed for second graph
    df["len_msg"] = df.message.str.len()
    col_names = []
    len_message = []
    for i in range(4, df.shape[1]-1, 1):
        len_message.append(df.loc[df[df.columns[i]] == 1]["len_msg"].mean())
        col_names.append(df.columns[i])

    # create the second graph
    graphs.append({
        'data': [
            Bar(
                x=col_names,
                y=len_message,
                orientation = 'v',
                width=.5,
                textfont = {'family' : 'Arial'},
                marker=dict(color='silver')
            )
        ],

        'layout': {
            'title': 'Average Message Length per Category',
            'yaxis': {
                'title': "Average Message Lenght"
            },
            'xaxis': {
                'title': "Category",
                'automargin': True
            },
            'padding': 150
        },

    })

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    '''
    as the main function this runs whenever the file is called

    it sets the port and then runs the app through the desired port
    '''

    if len(sys.argv) == 2:
        app.run(host='0.0.0.0', port=int(sys.argv[1]), debug=True)
    else:
        port = int(os.environ.get('PORT', 5000))
        app.run(host='0.0.0.0', port=port)



if __name__ == '__main__':
    main()