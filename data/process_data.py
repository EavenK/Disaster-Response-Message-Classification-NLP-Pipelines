# import libraries
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """ 
    Load messages and categories data files and 
    merged them into a single pandas dataframe
    
    # Parameters:
        messages_filepath (str): full filepath to messages csv file
        categories_filepath (str): full filepath to categories csv file
    
    # Returns:
        df (dataframe): merge of messages and categories data.
        
    """
    
    # load messages dataset and categories dataset
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = pd.merge(messages, categories, on='id')
    
    return df


def clean_data(df):
    """
    Split 'categories' column into 36 new columns with each column represents a category, 
    and convert them from string to binary. Clean dataframe by removing duplicates.	
    
    # Parameters:
        df (dataframe): Dataframe containing merged content of messages & categories datasets.
    
    # Return:
        df (dataframe): cleaned version of input dataframe
        
    """
    
    
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    
    # used the first row of the categories dataframe to extract a list of new column names for categories.
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x : x[:-2])
    
    # rename the columns of `categories`
    categories.columns = category_colnames

    # Iterate through the category columns in df to keep only the last character of each string (the 1 or 0).
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    #child_alone column is not needed as it is always 0. so drop
    if len(np.unique(categories['child_alone'])) == 1:
        categories.drop('child_alone', axis =1, inplace=True)        
    
    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1 )
    
    # drop duplicates
    df.drop_duplicates(inplace=True)
    
    # remove rows with value 2 from 'related' column 
    df = df[df['related'] != 2]
    
    

    return df
    
    

def save_data(df, database_filename):
    """
    Save the input dataframe `df` into a database 
    specify by 'databse_filename'
    
    #Parameters:
        df (dataframe): dataframe to save
        database_filename (str): Filename for output database
    
    """

    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('Messages', engine, index=False, if_exists='replace') 


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