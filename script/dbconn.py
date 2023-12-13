import pandas.io.sql as sqlio
import psycopg2
from psycopg2 import sql
from sqlalchemy import create_engine
import pandas as pd


def db_connection_sqlalchemy():

    engine = create_engine('postgresql+psycopg2://postgres:1234@localhost/telecom')

    return engine

def db_read_table_sqlalchemy(engine, table_name):
    query = f'SELECT * FROM {table_name}'
    df= pd.read_sql_query(query, engine)
    return df


def db_connection_psycopg():
    
    pgconn = psycopg2.connect(dbname="telecom",user="postgres",password="1234",host="localhost",port="5432")
    return pgconn


def db_read_table_psycopg(pgconn, table_name):
    sql = f'SELECT * FROM {table_name}'
    df = sqlio.read_sql_query(sql, pgconn)
    return df

def db_write_table_psycopg(pgconn, tablename, df):
    pass

def db_delete_table_pyscopg():
    cursor = pgconn.cursor()
    
    drop_table_query = sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(sql.Identifier(table_name))

    cursor.execute(drop_table_query)

    pgconn.commit()

    print(f"Table `{table_name}` has been successfully deleted.")

    if cursor:
        cursor.close()





