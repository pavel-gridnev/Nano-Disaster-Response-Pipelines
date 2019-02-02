# Disaster Response Pipeline Project
The project uses dataset from historical disaster events and uses machine learnig techniques to help 
categorize text messages and direct them to corresponding authorities during similar events in the 
future. 

### Installation
Clone repository from Github and execute pip install to deploy necessary packages from `requirements.txt` file

```
git clone https://github.com/pavel-gridnev/Nano-Disaster-Response-Pipelines.git
cd Nano-Disaster-Response-Pipelines
pip install -r reqirements.txt
```

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Project details

### ETL pipeline
1. Category `child_alone` is removed from dataset on ETL stage, it is empty.
2. Category `related` has rows with values=2. Those rows represent incomplete data and empty in other categories.
Removed from dataset.
3. Empty rows are removed from dataset too, model has nothing to learn from them. 
### ML pipeline
RandomForest calssifier is used for predictions. Grid search discovered set of the parameters that are used in the code 
for the model:
1. n_estimators=10
1. max_depth=500

### Evaluation results
Trained classification model exhibits good results for accuracy of around 80-90% 
for categories with sufficient amount of positive labels. Recall values shuld be improved significantly, 
because model is going to miss numerous important messages.

Several categories like `offer, security, missing_people, etc.` have very unbalanced data for predicted classes. 
Fot those categories I tried to employ `class_weight` parameters, however the the results didn't improve
and those categories still had zero accuracy and recall values.

### Extra tools
Use `improve_classifier.py` file to improve performnce of the model by performing extended GridSearches
or observing performance of the classifier with weights plugged in.
<p>Usage:

```
improve_classifier.py show_params
improve_classifier.py evaluate_category ../data/DisasterResponse.db offer
improve_classifier.py grid_search ../data/DisasterResponse.db classifier.pkl -v
```
Consult get_parser method for detailed command line parameters explanation.

 


