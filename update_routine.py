# Automatically add new information from new images into yout SQLite database
# Periodically checks for new images and updates the database accordingly
# Repeat the process at regular intervals or trigger


from sqlite_db import ImgTextStore

from detect_recogn_runner import main
import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from tqdm import tqdm
import sys

IMAGESTORAGE = '.\_IMAGESTORAGE'
DBSTORAGE = '.\_DB'
STATISTICS = '.\_STATISTICS'


class NullWriter():
    def write(self, txt):
        pass


class NewFileHandler(FileSystemEventHandler):
    def __init__(self, folder_path):
        super().__init__()
        self.folder_path = folder_path
        self.new_files = set()

    def on_created(self, file_event1):
        if file_event1.src_path.endwith(".png"):
            print("New file created ", file_event1.src_path)
            self.new_files.add(os.path.basename(file_event1.src_path))

# Let's do it later
def monitor_folder(imgs_folder):
    modification_time = time.ctime(os.path.getmtime(imgs_folder))
    print(modification_time)
    #existing_files = set(os.listdir(imgs_folder))
    #print(existing_files)

    #event_handler = NewFileHandler(imgs_folder)
    #observer = Observer()
    #observer.schedule(event_handler, imgs_folder, recursive=False)
    #observer.start()



def update_database(init_folder):
    db_name = 'DB_' + init_folder + '.db'
    db_name = os.path.join(DBSTORAGE, db_name)


    img_text_store = ImgTextStore(db_name)
    db_len = img_text_store.num_rows()
    print("Number of lines already in DB ", db_len)

    existing_ids = img_text_store.get_existing_image_ids()
    print(existing_ids)

    init_folder = os.path.join(IMAGESTORAGE, init_folder) # update the full path to images
    all_files = sorted(os.listdir(init_folder)) # Do I need read files sequentially?

    ocr_time_img = []
    print("Start process ")
    #sys.stdout = NullWriter() # disable all print statements

    t0_0 = time.time()
    for im in tqdm(all_files):
        img_id = os.path.basename(im).split('.')[0]
        if img_id not in existing_ids:

            #IMPORTANT value = "GET VALUE"
            full_path_im = os.path.join(init_folder, im)
            #print(full_path_im)
            t0 = time.time()
            #text_out = ReadImagesFromFolder(full_path_im)

            print("777 ", full_path_im)
            text_out = main(full_path_im)
            t1 = time.time() - t0
            ocr_time_img.append(str(round(t1,2)))

            if text_out: # it's not empty. Means this is the image HAS any text on it

                #print(text_out)
                #print("Time in sec OCR elapsed is ", time.time() - t0)

                t = ", ".join(text_out)

                img_text_store.insert_key_value(img_id, t)



    sys.stdout = sys.__stdout__
    db_len1 = img_text_store.num_rows()
    print("Number of lines added in DB ", db_len1 - db_len)
    img_text_store.close_db() # re-enable printing
    t0_1 = time.time() - t0_0
    print("DB updated in ", t0_1, " sec")

    #with open("Ocr_time_img.txt", "w") as outfile:
    #    outfile.write("\n".join(ocr_time_img))

    #last_row = img_text_store.get_row(db_len-1)
    #print(last_row)

    #scan_folder(init_folder)





if __name__=="__main__":
    #init_folder = "remained\kdukalis"
    #db_name = "key_value_store2.db"

    init_folder = "10_Oct"

    update_database(init_folder)


