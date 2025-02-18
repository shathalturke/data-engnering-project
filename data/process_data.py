import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    This load_data function:
    takes paths to messages and categories CSV files, processes the data, and returns a merged dataframe with cleaned categories.
    '''
    # load dataset
    categories = pd.read_csv(categories_filepath)
    messages = pd.read_csv(messages_filepath)
    # merge datasets
    df = messages.merge(categories, on='id', how='inner')
    # create a dataframe of the 36 individual category columns
    categories = df["categories"].str.split(';', expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0,:]
    # use this row to extract a list of new column names for categories.
    category_colnames = row.apply(lambda x: x[:-2])
    # rename the columns of `categories`
    categories.columns = category_colnames
        # Convert category values to just numbers 0 or 1.
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    categories.replace(2, 1, inplace=True)
    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace = True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    return df


def clean_data(df):
    '''
    This clean_data function removes duplicate rows from the input dataframe df and returns the cleaned dataframe without duplicates.
    '''
    # drop duplicates
    df = df.drop_duplicates()
    return df
    

def save_data(df, database_filename):
    '''
    This save_data function saves the input dataframe df to an SQLite database specified by database_filename with table name 'DisasterResponse', replacing existing data.
    '''
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('DisasterResponse', engine,if_exists = 'replace', index=False)  


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
