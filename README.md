# Disaster Response Pipeline Project
In this repository I share the code underlying the text-classifier web app that I published here:
> https://disaster-nlp-app.herokuapp.com

As you can see from the web-app, the aim of this project is to build a text classsifier to detect disaster messages.

![](app/templates/static/ss_app.png)

## Summary
Be it man-made or caused by natural reasons, sadly, disasters are fundamental part of human life. We can't eliminate them
all together. However, what we can do is reacting to them smartly. One step towards achieving smart action is reducing the 
response time. As response time can decide life and death under such circumstances, being fast and proactive is vital.

From the accuracy perspective, human intuition is still unmatched by our state-of-the-art algorithms. But human intuition 
comes with its price. It's not guaranteed to be fast enough and, more importantly it's not scalable. When a disaster 
happens we can't be sure that we'll have enough workforce to distinguish distress calls from regular interactions. 
But algorithms can serve us 24/7 and can cope with huge amount of data. For this reason I tried to build a  web-app which
can take any ext and tell us whether it's disaster related or not. 

As you'll read in the Remarks section, current version of this app has severe limitations both from the accuracy and 
scalability perspective. Yet, this is a step forward. In ideal settings, an app like this can help local authorities 
proactively help people in disaster conditions. Automatic detection of disaster situations and sendin immediate alerts 
to the authorities not only reduce response time, it also increases the proactivity. Aid comes to you before you reach 
the authorities just because some people tweeted about it, wouldn't you like that?
 

## Structure of the Repository
```
app
| - template
| | | - static # directory that contains images
| |- master.html # main page of web app
| |- go.html # classification result page of web app
|- run.py # Flask file that runs app
data
|- disaster_categories.csv # data to process
|- disaster_messages.csv # data to process
|- process_data.py # file to handle ETL
|- DisasterResponse.db # database to save clean data to
models
| | | - catboost_info # training output of CatBoostClassifier
|- train_classifier.py # file to handle NLP pipelines
|- classifier.pkl # saved model
README.md
nltk.txt # nltk dependencies for heroku. No need for local
Procfile # directions for Heroku to find the web-app
```

## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database, first cd into data directory.
   
      >`python process_data.py '\
      'disaster_messages.csv disaster_categories.csv '\
      'DisasterResponse PostEtl`
   
    - To run ML pipeline that trains classifier and saves, first cd into models directory.
      > `python train_classifier.py ../data/DisasterResponse.db PostETL ClassifierName.pkl`

2. Run the following command in the app's directory to run your web app.

   >`python run.py`

3. Go to http://0.0.0.0:3001/


## Remarks and References

### Underfitting due to Poor Training 
My main motivation behind this project is to demonstrate my **pipelining skills** (from ETL to deployment) rather than
building a strong predictive model. For this reason, the **Catboost classifier** was trained with really low iterations.
For the same reason, I kept the parameter grid for the **GridSearchCV** really narrow. With more iterations and better 
grid search optimization, the classifier performance can be increased. However, this requires investing more time on 
training. Nevertheless, if more time and computational power are available, then, I would pick any model listed on
hugginface.co over catboost classifier. 

### Underfitting due to Class Imbalance
The categories are not equally represented in the dataset. There are even some categories with no observations. This is 
called class imbalance. In this project, I took no action to counter class imbalance problem. In the mockup phase, I 
attempted to use SMOTE from imbalanced learn library. But both sklearn pipeline and encoding of the target (multilabel) 
was creating incompatibilities and for this reason I decided not to continue with this line.

### Scalability
As this is a demo-app, I used heroku to quickly deploy it. The negative part of Heroku is its free tier limitations. The
app loads slowly and it wouldn't be reliable if more people tries to access it at the same time. 

### Reference
The plotly script was not visible on my app. I tried my own solutions but couldn't figure out the reason by myself. I 
learned the solution to this problem at
>
> https://github.com/quantumphysicist/Disaster-Response-Pipelines/blob/main/app/templates/master.html

I used the the repo above as reference for my visualisations aesthetics as well. I would like to thank the author. 

The second problem I faced was deploying the app to **Heroku**. I learned about nltk.txt file from the Heroku error 
messages, however, even after fixing the nltk issues, the was not running. I learned about the proper procfile & 
def main() configurations from this repository.
>
> https://github.com/madkehl/DisasterResponse/tree/main/web_app
