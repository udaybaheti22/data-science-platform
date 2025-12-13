from .datastore import data_store
import datetime

def log_action(description: str):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data_store["logs"].insert(0, {"timestamp": timestamp, "description": description})

