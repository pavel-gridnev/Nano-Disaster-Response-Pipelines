import sys
import nltk
nltk.download(['punkt', 'wordnet'], quiet=True)
from sqlalchemy import create_engine
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report
from sklearn.externals import joblib


def load_data(database_filepath, table_name='DisasterResponseTable') :
    """
    Loading data from table in SQLlite database
    :param database_filepath: source database file name
    :param table_name: Table name to load data from, defaults to  DisasterResponseTable
    :return:
        X - numpy.ndarray of original text
        Y - numpy.ndarray of categories labels
        category_names - textual categories list
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table(table_name, engine)

    X = df['message'].values
    y = df.iloc[:, 4:].values
    category_names = df.columns[4:]
    return X, y, category_names


def tokenize(text):
    """
    Tokenize free text by lower casing text, splitting into words, lemmatazing separate words
    :param text: free text
    :return: list of tokens
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens]
    return clean_tokens


def build_model():
    """
    Builds pipeline for training RandomForestClassifier composed of CountVectorizer and TfidfTransformer as transformers
    In simple words, converts text to matrix and assigns weight for each word in the matrix
    :return: pipeline object
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=10, max_depth=500), n_jobs=1))
    ])

    return pipeline


def build_pipeline_for_category(class_weight=None, n_estimators=10, max_depth=500):
    """ Builds pipeline for particular category
    :return: pipeline object
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', RandomForestClassifier(class_weight=class_weight, n_estimators=n_estimators, max_depth=max_depth))
    ])

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """ Prints out classification report: Accuracy, Recall for each label in each category predicted using text data

    :param model: model to evaluate, can be either SearchGrid or Pipeline estimator
    :param X_test: text data to test
    :param Y_test: true labels to compare to predictions
    :param category_names: list of the categories to be predicted
    :return: None
    """
    if isinstance(model, GridSearchCV):
        # it is cv grid
        pipeline = model.best_estimator_
    else:
        pipeline= model

    y_pred = pipeline.predict(X_test)
    if len(y_pred.shape)==1:
        y_pred = y_pred.reshape(-1,1)
        Y_test = Y_test.reshape(-1, 1)

    for idx, category in enumerate(category_names):
        print(f'Classification report for: {category}')
        cr = classification_report(y_true=Y_test[:, idx], y_pred=y_pred[:, idx], output_dict=True)
        # remove avg keys from report
        [cr.pop(key) for key in ['micro avg', 'macro avg', 'weighted avg']];
        crs = [{'class_label': item[0], **item[1]} for item in cr.items()]
        print(pd.DataFrame.from_dict(crs).to_string(index=False))
        print()


def save_model(model, model_filepath):
    """
    Stores model file as pickle file in the local folder
    In case of GridSearch the best estimator is saved
    :param model: model object
    :param model_filepath: pickle filename to store with
    :return: None
    """
    if isinstance(model, GridSearchCV):
        joblib.dump(model.best_estimator_, model_filepath)
    else:
        joblib.dump(model, model_filepath)

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
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()