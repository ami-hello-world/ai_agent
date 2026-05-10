import sqlite3

class chatDB:
    def __init__(self,db_path,table_name):
        self.db_path = db_path
        self.table_name = table_name
        self.create_table()

    def create_table(self):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute( f'''
        CREATE TABLE IF NOT EXISTS {self.table_name}(
        id INTEGER PRIMARY KEY,
        session_id INT NOT NULL,
        timestamp TEXT NOT NULL,
        dialogue TEXT,
        summary TEXT,
        vector BLOB
        )
        '''
        )
        conn.commit()
        conn.close()

    def commit_db(self,session_id,timestamp,dialogue,summary=None,vector=None):
        data = (session_id,timestamp,dialogue,summary,vector)
        cur = conn.cursor()
        cur.execute( 'INSERT INTO chatlog(session_id, timestamp, dialogue, summary, vector)VALUE(?,?,?,?,?)',data)
        conn.commit()
        conn.close()
