
import pandas as pd
from script import dbconn

engine_conn = dbconn.db_connection_sqlalchemy()

#function used to load data from PostgreSQl using dbconn module.

def load_data_from_postgres():
    pgconn = dbconn.db_connection_psycopg()
    return dbconn.db_read_table_psycopg(pgconn, 'xdr_data')

#function used to load data from SQLAlchemy using dbconn module.

def load_data_from_sqlalchemy():

    return dbconn.db_read_table_sqlalchemy(engine_conn, 'xdr_data')

#function used to clear the null values and preprocess the DataFrame.

def clear_data(df):

    df_clean = df[['Bearer Id', 'Start']]

    df_clean['Start'] = pd.to_datetime(df_clean['Start'])

    return df_clean

#function that will give a through description of the columns and their types

def describe_data(df_clean_data):
    data_description = df_clean_data.dtypes.reset_index()
    data_description.columns = ['Variable', 'Data Type']
    
    return data_description

#function used to read the sql data into a DataFrame

def read_data():
 
    return pd.read_sql_table('clean_data', engine_conn)