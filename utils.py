import json

def load_dictionary_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
        dictionary = json.load(file)
    return dictionary

