import pandas as pd
from sqlalchemy import create_engine
import sys


def load_data(messages_filepath, categories_filepath):
    """
    This function merges two dataframes and creates a
    dataframe contains messages and categories.

    INPUT:
    messages_filepath: file path for the 'messages' file
    categories_filepath: file path for the 'categories' file

    OUTPUT:
    Pandas Dataframe
    """

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    df = pd.merge(messages, categories, on='id')
    return df


def clean_data(df):
    """
    Helper function to clean the 'df'

    INPUT:
    df: pandas dataframe

    OUTPUT:
    Cleaned version of 'df'
    """
    categories = df['categories'].str.split(';', expand=True)

    row = categories.iloc[0]
    category_colnames = row.apply(lambda cat: cat[:len(cat) - 2])
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].astype(str).apply(lambda col: col[-1])
        categories[column] = pd.to_numeric(categories[column])
    categories.replace(2, 1, inplace=True)
    df.drop(columns='categories', inplace=True)

    df = pd.concat([df, categories], axis=1)
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    """
    Function to save the cleaned 'df' as a database.

    INPUT:
    df: Pandas Dataframe to be saved
    database_filename: Name of database
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('CleanMessages', engine, index=False)


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
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')


if __name__ == '__main__':
    main()




