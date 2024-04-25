import json
import sqlite3

class ImgTextStore:
    def __init__(self, db_file):
        self.db_file = db_file
        self.conn = sqlite3.connect(db_file)  # Use ':memory:' for in-memory database
        self.cursor = self.conn.cursor()  # In order to execute SQL statements and fetch results from SQL queries
        self._initialize_database()

    def _initialize_database(self):
        self.cursor.execute("CREATE TABLE IF NOT EXISTS hash_map(img_id, text)")  # Create the table hash_map1
        self.conn.commit()

        # Fetch the resulting row
        self.cursor.execute("SELECT name FROM sqlite_master")
        #print("Name of the table in db ", self.cursor.fetchone())

    def insert_key_value(self, id, id_text):
        self.cursor.execute("INSERT INTO hash_map VALUES (?, ?)", (id, id_text))  # Insert key-value pairs into the table
        self.conn.commit()

    def get_existing_image_ids(self):
        """
        Retrieve existing image IDs from the database.
        """
        self.cursor.execute("SELECT img_id FROM hash_map")
        existing_ids = [row[0] for row in self.cursor.fetchall()]
        return existing_ids



    def search_by_value(self, query):
        self.cursor.execute("SELECT img_id FROM hash_map WHERE text LIKE ?", ('%' + query + '%',))
        res = [row[0] for row in self.cursor.fetchall()]
        # print(res)
        return res

    def get_row(self,indx):
        self.cursor.execute("SELECT * FROM hash_map LIMIT 1 OFFSET ?", (indx,))
        # Fetch the result
        indx_row = self.cursor.fetchone()
        return indx_row

    def num_rows(self):
        self.cursor.execute("SELECT COUNT(*) FROM hash_map")
        row_count = self.cursor.fetchone()[0]
        #print("Num of rows in db ", row_count)
        return row_count

    def search_by_id(self, img_id):
        self.cursor.execute("SELECT text FROM hash_map WHERE img_id LIKE ?", ('%' + img_id + '%',))
        res = [row[0] for row in self.cursor.fetchall()]
        return res

    def close_db(self):
        self.conn.close()




if __name__ == "__main__":
    json_file = 'remained_images.json'

    with open(json_file, encoding='utf-8') as f:
        d = json.load(f)
    #print(d.keys())

    #print(d['IMG_0828.PNG'].split())
    #datas = {k: v[0].split() for k, v in d.items()}
    #print(datas['IMG_0828.PNG'])
    #print(d['IMG_0828.PNG'])

    db_name = "key_value_store1.db"
    img_text_store = ImgTextStore(db_name)

    # Insert img_id - text pairs
    if img_text_store.num_rows() == 0:
        for key, value in d.items():
            t = ", ".join(value)
            img_text_store.insert_key_value(key, t)

    row_val = img_text_store.get_row(3)
    print(row_val)

    # Search for img_id's whose values contain the substring query
    res = img_text_store.search_by_value("elevator")
    print(res)

    img_text_store.close_db()
