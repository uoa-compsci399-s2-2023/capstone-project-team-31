import argparse
import json
import sqlite3

def add_images_from_json(json_file, db_path):
    """
    Add images to the database from a json file
    :param json_file: path to the json file
    :return: None
    """
    # Open the json file
    try:
        f = open(json_file, "r")
        data = dict(json.load(f))
        f.close()
    except Exception as e:
        print("Error in opening images json file.")
    
    # Connect to the database
    try:
        conn = sqlite3.connect(db_path)
    except Exception as e:
        print("Error in connecting to the database.")
    
    #
    
    for img in data.keys():
        add_image_to_db(data[img], conn)
        
def add_image_to_db(data:dict, conn: sqlite3.Connection):
    """
    Add an image to the database
    data: dict containing the image information
    conn: sqlite3.Connection object
    :return: None
    """
    cursor = conn.cursor()
    # Checks if the image is already in the database
    cursor.execute("SELECT * FROM images WHERE img_base64=?", (data["img_base64"],))
    if cursor.fetchone() is None: # If the image is not in the databse, add it to the database
        cursor.execute("INSERT INTO images VALUES (?,?,?,?,?)", (data["img_base64"], data["ethinicity"], data["gender"], data["age"], data["emotion"]))
        conn.commit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Add images to the database')

    parser.add_argument('--db', default=None, type=str, help='Path to the images')
    parser.add_argument("--imgs", default=None, type=str, help="Path to the json file containing images information")

    args = parser.parse_args()    

    add_images_from_json(args.imgs, args.db)