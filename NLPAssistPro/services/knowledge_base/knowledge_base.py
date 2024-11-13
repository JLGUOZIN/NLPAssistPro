import sqlite3

class KnowledgeBase:
    def __init__(self):
        self.conn = sqlite3.connect('knowledge_base.db')
        self.create_table()

    def create_table(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS faq (
                question TEXT,
                answer TEXT
            )
        ''')
        self.conn.commit()

    def add_entry(self, question, answer):
        cursor = self.conn.cursor()
        cursor.execute('INSERT INTO faq (question, answer) VALUES (?, ?)', (question, answer))
        self.conn.commit()

    def query(self, question):
        cursor = self.conn.cursor()
        cursor.execute('SELECT answer FROM faq WHERE question = ?', (question,))
        result = cursor.fetchone()
        return result[0] if result else None
