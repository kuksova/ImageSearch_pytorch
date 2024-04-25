import argparse

from sqlite_db import ImgTextStore
import os
import shutil
import time
import json

# Base path
IMAGESTORAGE = '..\_IMAGESTORAGE'
DBSTORAGE = '..\_DB'
STATISTICS = '..\_STATISTICS'

def clean_copies():
    """
    Copies that returned by keyword. Just for conviniet showing to user.
    """
    shutil.rmtree("_SEARCHRESULTS")



def get_images_by_text(init_folder, db_name, query_text):
    init_folder = os.path.join(IMAGESTORAGE, init_folder)
    # 1. Step

    if os.path.exists(".\_SEARCHRESULTS"):
        clean_copies()

    print(db_name)
    img_text_store = ImgTextStore(db_name)

    # Search for img_id's whose values contain the substring query
    res = img_text_store.search_by_value(query_text)
    img_text_store.close_db()

    print("Founded images ", len(res), "with query - ", query_text)
    print("You can find them in _SEARCHRESULTS.")

    # Save Id's for found images for accuracy and statistics
    txt_name = os.path.join(STATISTICS, "accuracy.txt")
    with open(txt_name, "w", encoding="utf-8") as outfile:
        print(res)
        outfile.write('\n')
        outfile.write(query_text+' ')
        outfile.write(str(len(res))+' ')
        outfile.write(' '.join(res))
    json_name = os.path.join(STATISTICS, "accuracy.json")
    data = {"query": query_text, "Nimages": len(res), "foundIDs":res}
    with open(json_name, 'w') as json_file:
        json.dump(data, json_file)

    temp_dir = os.makedirs("_SEARCHRESULTS") #Create a temporary copy of images in a temporary folder that has name as a query

    for im_id in res:
        full_path_im = os.path.join(init_folder, im_id+'.jpg')
        temp_image_path = os.path.join(".\_SEARCHRESULTS" , im_id+'.jpg')

        shutil.copy(full_path_im, temp_image_path)  # Copy the image file to the temporary directory


def check_db(db_name):
    #query_text = "mironovanastasiia"
    #query_text = "harrynuriev"
    query_text = "harrynuriev"
    print("666 ", db_name)
    img_text_store = ImgTextStore(db_name)
    print(img_text_store.num_rows())
    for i in range(8):
        b = img_text_store.get_row(i)
        print(b)

    #c = img_text_store.search_by_id('01a00e5f287a2c23f6ece646aa00bf08d5c6da0c32')
    c = img_text_store.search_by_id('01a28453a0a988b3290d5f746c882578d66574d432')
    print("Search by ID", c)

    res = img_text_store.search_by_value(query_text)
    if not res:
        print("No images found for this keyword!")
    else:
        print("FOUNDED IMAGES ", res)

    #
    idf = img_text_store.get_existing_image_ids()
    print(idf)


    img_text_store.close_db()


def main(query_text, init_folder='all'):
    #parser = argparse.ArgumentParser(description='Search Images')
    #parser.add_argument('--key-word', type=str, default=[], help='type the name of your favourite blogger or topic ')
    #parser.add_argument('--folder', type=str, default='all')

    #args = parser.parse_args()

    #if args.folder == 'all': # Search for in all DB folders

    if init_folder=='all': # Search for in all DBs, return copies of images from whole image storage
        all_dbs = sorted(os.listdir(DBSTORAGE))
        t0 = time.time()
        for db_name in all_dbs:
            db_name = os.path.join(DBSTORAGE, db_name)
            # check_db(db_name)
            get_images_by_text(init_folder, db_name, query_text)

        print("Search time was ", time.time() - t0)


    else: # Search for in specific DB if the user knows exact month when he wants to search
        db_name = 'DB_'+ init_folder +'.db'
        print(db_name)
        db_name = os.path.join(DBSTORAGE, db_name)
        t0 = time.time()
        #check_db(db_name)
        get_images_by_text(init_folder, db_name, query_text)
        print("Search time was ", time.time() - t0)





if __name__ == "__main__":

    query_text = 'mironova'

    init_folder = '09_Sep'

    main(query_text, init_folder)



