import sys
import os
sys.path.insert(0, '/home/Kdixter/Desktop/DSM_Final_Project/final_project/src/data_gathering')
from importlib import import_module
loader = import_module('07_load_data_to_db')

if __name__ == "__main__":
    from sqlalchemy.orm import Session
    with Session(loader.engine) as session:
        try:
            loader.load_news_corpus(session)
        except Exception as e:
            print("\n--- EXACT DATABASE ERROR ---")
            print(str(e).split("[SQL:")[0])
