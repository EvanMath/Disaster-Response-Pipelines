import pandas as pd
from sqlalchemy import create_engine


messages = pd.read_csv('messages.csv')
categories = pd.read_csv('categories.csv')

df = pd.merge(messages, categories, on='id')

categories = df['categories'].str.split(';', expand=True)

row = categories.iloc[0]
category_colnames = row.apply(lambda cat: cat[:len(cat) - 2])
categories.columns = category_colnames
for column in categories:
    categories[column] = categories[column].astype(str).apply(lambda col: col[-1])
    categories[column] = pd.to_numeric(categories[column])

df.drop(columns='categories', inplace=True)

df = pd.concat([df, categories], axis=1)
df.drop_duplicates(inplace=True)

engine = create_engine('sqlite:///CleanMessages.db')
df.to_sql('cleanMessages', engine, index=False)