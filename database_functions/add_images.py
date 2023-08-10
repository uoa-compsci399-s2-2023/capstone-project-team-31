import argparse
import json
import sqlite3
import base64
import cv2

def add_images_from_json(json_file:str, db_path:str):
    """
    Add images to the database from a json file
    
    Args:
        param json_file: path to the json file
        param db_path: path to the database
    """
    # Open the json file
    try:
        f = open(json_file, "r")
        data = dict(json.load(f))
        f.close()
    except Exception as e:
        print("Error in opening images json file.")
        exit()
    
    # Connect to the database
    try:
        conn = sqlite3.connect(db_path)
    except Exception as e:
        print("Error in connecting to the database.")
        exit()
    
    # Loop through json file
    for img in data.keys():
        # Convert the image to base64
        img_b64 = convert_to_b64(img)
        # Add the image to the database
        add_image_to_db(img_b64, data[img], conn)
        
def add_image_to_db(img_b64:str, tags:dict, conn: sqlite3.Connection):
    """
    Add an image to the database
    
    Args:
        img_b64: base64 string of the image
        tags: dict containing the image tags
        conn: sqlite3.Connection object

    """
    cursor = conn.cursor()
    # Checks if the image is already in the database
    cursor.execute("SELECT * FROM images WHERE img_base64=?", (img_b64,))
    if cursor.fetchone() is None: # If the image is not in the databse, add it to the database
        cursor.execute("INSERT INTO images VALUES (?,?,?,?,?)", (img_b64, tags["ethinicity"], tags["gender"], tags["age"], tags["emotion"]))
        conn.commit()
        
def convert_to_b64(img_dir:str):
    """
    Convert an image to base64
    
    Args:
        img_dir (str): Path to the image (only .jpg, .jpeg and .png are supported)

    Returns:
        img_b64: base64 string of the image
    """
    # Read the image
    img = cv2.imread(img_dir)
    
    # Check the image format
    if img_dir.endswith(".jpg") or img_dir.endswith(".jpeg"): 
        _, img_b64 = cv2.imencode('.jpg', img)
    elif img_dir.endswith(".png"):
        _, img_b64 = cv2.imencode('.png', img)
    else:
        print("Error in image format.")
        exit()
        
    # Convert the image to base64
    img_b64 = str(base64.b64encode(img_b64))[2:-1]
    return img_b64

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Add images to the database')

    parser.add_argument('--db', default=None, type=str, help='Path to the images')
    parser.add_argument("--imgs", default=None, type=str, help="Path to the json file containing images information")

    args = parser.parse_args()    

    add_images_from_json(args.imgs, args.db)