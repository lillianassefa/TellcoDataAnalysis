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

def marketing_recommendation(df,self):
    interpretation = "Based on the user overview analysis, the marketing team should focus on:"
    
    # Top handsets
    top_handsets_list = top_handsets(df).index.tolist()
    interpretation += f"\n- Promoting the top handsets: {', '.join(top_handsets_list)}"

    # Top manufacturers
    top_manufacturers_list = top_manufacturers(df).index.tolist()
    interpretation += f"\n- Collaborating with the top manufacturers: {', '.join(top_manufacturers_list)}"

    # Top handsets per manufacturer
    for manufacturer in top_manufacturers_list:
        top_handsets_list = top_handsets_per_manufacturer(df, manufacturer).index.tolist()
        interpretation += f"\n- Highlighting top handsets for {manufacturer}: {', '.join(top_handsets_list)}"

    return interpretation




