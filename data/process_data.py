import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath:str, categories_filepath:str):
    """
    Data loader function that takes the filepaths as params. Merges them and  returns the merged dataframe
    :param messages_filepath: filepath of the 'messages.csv'
    :param categories_filepath: filepath of the 'categories.csv'
    :return: merged dataframe
    """
    messages = pd.read_csv(messages_filepath, encoding="utf-8")
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, how="left", on="id")
    return df


def clean_data(df):
    """
    A custom unction to clean dataframe following the list of operations below:

    1) Pick categories column of df. Split each row of it by ";" separator. Expand it. Store it under categories
    dataframe.
    2) Delete last two characters of the first row of each column in the categories dataframe. Assign them as column
    names.
    3) Delete the non-numerical characters in each row of the categories dataframe and store them as integers.
    4) Return to original dataframe and drop the categories column. Then, concat original dataframe and categories
    dataframe.
    5) Concatenate original dataframe and categories dataframe
    6) Drop duplicates

    :param df: Dataframe to be cleaned
    :return: clean dataframe
    """

    # Step 1
    categories = df["categories"].str.split(";", expand=True)
    row = categories.iloc[0]
    # Step 2
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames

    # Step 3
    for column in categories:
        categories[column] = categories[column].apply(lambda row: row[-1])
        categories[column] = categories[column].astype("int")


    # Step 4
    df.drop(columns=["categories"], inplace=True)

    # Step 5
    df = pd.concat([df, categories], axis=1)

    # Step 6
    df.drop_duplicates(inplace=True)

    # My personal exploration showed me that there are non-binary data left in 'related' column.
    df = df[df['related']!=2]

    return df


def save_data(df, database_filepath, table_name):
    engine = create_engine('sqlite:///{}.db'.format(database_filepath))
    df.to_sql('{}'.format(table_name), engine, index=False, if_exists="replace")


def main():
    if len(sys.argv) == 5:

        messages_filepath, categories_filepath, database_filepath, table_name = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {} to table: {}'.format(database_filepath, table_name))
        save_data(df, database_filepath, table_name)
        
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