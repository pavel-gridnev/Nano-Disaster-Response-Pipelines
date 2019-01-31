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


def load_data(database_filepath, table_name='DisasterResponseTable'):
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table(table_name, engine)

    X = df['message'].values
    y = df.iloc[:, 4:].values
    category_names = df.columns[4:]
    return X, y, category_names


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens]
    return clean_tokens


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=10, max_depth=500), n_jobs=1))
    ])

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    if isinstance(model, GridSearchCV):
        # it is cv grid
        pipeline = model.best_estimator_
    else:
        pipeline= model

    y_pred = pipeline.predict(X_test)
    for idx, category in enumerate(category_names):
        print(f'Classification report for:{category}')
        print(classification_report(y_true=Y_test[:, idx], y_pred=y_pred[:, idx]))


def save_model(model, model_filepath):
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