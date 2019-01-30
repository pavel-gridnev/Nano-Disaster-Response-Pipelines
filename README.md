# Disaster Response Pipeline Project
The project uses dataset from historical disaster events and uses machine learnig techniques to help 
categorize text messages and direct them to corresponding authorities during similar events in the 
future. 

### Installation:
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

### ETL process
1. Category `child_alone` is removed from dataset on ETL stage, it is empty.
### ML pipeline
### Data analysis notes

