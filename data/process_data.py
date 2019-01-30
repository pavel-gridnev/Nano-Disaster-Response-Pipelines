import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Loading data from local CSV files and merge messages with their corresponding categories
    :param messages_filepath: source file with messages
    :param categories_filepath: source file with categories
    :return: merged dataframe
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how='left', left_on='id', right_on='id')
    return df

def clean_data(df):
    """
    Clean data in source dataframe:
    1. Split categories into separate columns
    2. Decode column values into binary values 0 or 1
    3. Remove duplicate rows
    :param df: source df obtained from load_data function
    :return: cleaned dataframe
    """
    categories = df['categories'].str.split(';', expand=True)
    category_colnames = categories.iloc[0, :].str.split('-', expand=True)[0]
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.slice(-1).astype(int)

    df.drop(['categories'], axis=1, inplace=True)
    cleaned_df = pd.concat([df, categories], axis=1)

    cleaned_df.drop_duplicates(inplace=True)

    constant_columns = [column for column in cleaned_df.columns[4:] if len(cleaned_df[column].unique().tolist()) == 1]
    if len(constant_columns):
        cleaned_df.drop(constant_columns, axis=1, inplace=True)

    assert cleaned_df.duplicated().sum() == 0, "Data frame has duplicates"
    return cleaned_df

def save_data(df, database_filename, table_name='DisasterResponseTable'):
    """
    Store data from data frame in the sqlite database
    :param df:
    :param database_filename: file name should have db extension

    """
    if database_filename[-3:] != '.db':
        raise ValueError('Wrong database file extension', database_filename)

    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql(table_name, engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()