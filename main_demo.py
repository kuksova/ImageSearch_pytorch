import PySimpleGUI as sg
import os
from PIL import Image
import io

from main_search import main

FIX_SHOW_IMAGES = 9

def search_in_specificDB(keyword, init_folder):
    # Search for in specific DB if the user knows exact month when he wants to search
    main(keyword, init_folder)

#sg.Window(title="Hello World", layout=[[]], margins=(100, 50)).read()
#layout = [[sg.Text("Hello from PySimpleGUI")], [sg.Button("OK")]]
def win():
    file_list_column = [
        [
            sg.Text("Image Folder"),
            sg.In(size=(25, 1), enable_events=True, key="-FOLDER-"),
            sg.FolderBrowse(),
        ],
        [
            sg.Listbox(
                values=[], enable_events=True, size=(40, 20), key="-FILE LIST-"
            )
        ],
    ]

    # For now will only show the name of the file that was chosen
    image_viewer_column = [
        [sg.Text("Choose an image from list on left:")],
        [sg.Text(size=(40, 1), key="-TOUT-")],
        [sg.Image(key="-IMAGE-")],
    ]

    # ----- Full layout -----
    layout = [
        [
            sg.Column(file_list_column),
            sg.VSeperator(),
            sg.Column(image_viewer_column),
        ]
    ]

    window = sg.Window("Image Viewer", layout)

    # Run the Event Loop
    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        # Folder name was filled in, make a list of files in the folder
        if event == "-FOLDER-":
            folder = values["-FOLDER-"]
            try:
                # Get list of files in folder
                file_list = os.listdir(folder)
            except:
                file_list = []

            fnames = [
                f
                for f in file_list
                if os.path.isfile(os.path.join(folder, f))
                   and f.lower().endswith((".png", ".jpg"))
            ]
            window["-FILE LIST-"].update(fnames)
        elif event == "-FILE LIST-":  # A file was chosen from the listbox
            try:
                filename = os.path.join(
                    values["-FOLDER-"], values["-FILE LIST-"][0]
                )
                window["-TOUT-"].update(filename)
                window["-IMAGE-"].update(filename=filename)

            except:
                pass

    window.close()




#win()


def jpg_to_png_resize(img_path, i):
    img = Image.open(img_path)
    # img_resized = img.resize((40, 60))
    base_width = 60
    wpercent = (base_width / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img_resized = img.resize((base_width, hsize), Image.Resampling.LANCZOS)

    temp_image_path = f"temp_image_{i}.png"
    img_resized.save(temp_image_path)
    return temp_image_path

def output_imgs(img_path, i):
    img = Image.open(img_path)
    # img_resized = img.resize((40, 60))
    base_width = 150
    wpercent = (base_width / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img_resized = img.resize((base_width, hsize), Image.Resampling.LANCZOS)

    temp_image_path = f"temp_image_{i}.png"
    img_resized.save(temp_image_path)

    # Create sg.Image element for the image
    image_element = sg.Image(filename=temp_image_path, key=f"-IMAGE-{i}")

    return temp_image_path, image_element

def get_images_grid(folder_path):
    image_extensions = [".png", ".jpg", ".jpeg"]
    image_files = [f for f in os.listdir(folder_path) if
        os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(tuple(image_extensions))]
    return image_files


def nm():
    layout1 = [
        [sg.Text("Hello from Image Search App", justification='left')],
        [
            sg.Text("Image Folder", justification='left'),
            sg.In(size=(25, 1), enable_events=True, key="-FOLDER-"),
            sg.FolderBrowse(),
        ],
        [sg.Text('Search'), sg.InputText(size=(20, 1), key="-TEXT-"), sg.Button("Search")],

        #[sg.Image(key="-IMAGE-", size=(6, 4))], # doesn't read .jpg
        #[sg.Image(key=f"-IMAGE-{i}", size=(20, 20), pad=(5, 5))
        #for i in range(9)]
        [sg.Image(key=f"-IMAGE-{i}", size=(20, 20), pad=(5, 5))
        for i in range(3) ],
        [
        sg.Image(key=f"-IMAGE-{i}", size=(20, 20), pad=(5, 5))
        for i in range(3, 6)
        ],
        [
        sg.Image(key=f"-IMAGE-{i}", size=(20, 20), pad=(5, 5))
        for i in range(6, 9)
        ],
    ]
    #layout2 = [
    #    [sg.Image(key="-IMAGE-")],]
        #for i in range(9)] ]

    layout3 = [
        #[sg.Text("The text:"],
        [sg.Text("The text:", justification='left'), sg.Text(size=(40, 1), key="-TOUT-")],
        #[sg.Column([], key="-IMAGES-", scrollable=True, size=(400, 400))]
        [sg.Image(key="-IMAGES-")],
        #[sg.Image(key=f"-IMAGE-{i}", size=(20, 20), pad=(5, 5)) for i in range(FIX_SHOW_IMAGES)]
    ]
    layout_general = [[
            sg.Column(layout1),
            #[sg.Column(layout1, element_justification="left", pad=(50, 10))],
            #sg.Column(layout2),
            sg.VSeperator(),
            sg.Column(layout3)]
            #[sg.Column(layout3, element_justification="right", pad=(0, 0))],
    ]

    window = sg.Window(title="Image Search", layout=layout_general, element_justification="center", size=(715, 515))

    while True:  # Event loop
        event, values = window.read()

        if event == sg.WINDOW_CLOSED:
            break
        name_db = 0
        if event == "-FOLDER-":
            folder = values["-FOLDER-"]
            name_db = os.path.basename(os.path.normpath(folder))
            window["-TOUT-"].update(name_db)

            images = get_images_grid(folder)
            try:
                for i, image_file in enumerate(images[:9]):

                    img_path = os.path.join(folder, image_file)
                    temp_image_path = jpg_to_png_resize(img_path, i)

                    #if img_path.lower().endswith((".jpeg", ".jpg")):
                    #    png_img = jpg_to_png(img_path=img_path)
                    #    print(type(png_img))
                    #    window["-IMAGE-"].update(data=png_img)

                    #if img_path.lower().endswith( ".png"):
                    window[f"-IMAGE-{i}"].update(filename=temp_image_path)
                    os.remove(temp_image_path)
            except:
                print(f"Error loading image: {img_path}")

        if event == "Search":
            query_text = values["-TEXT-"]
            folder = values["-FOLDER-"]
            name_db = os.path.basename(os.path.normpath(folder))
            print(folder)

            search_in_specificDB(query_text, name_db)

            # Show founded images in the window

            found_imgs = os.listdir("./_SEARCHRESULTS/")

            if not found_imgs or not query_text:
                print("No images found with search text")
                window["-TOUT-"].update(f"Founded images {0} with query - {query_text}") # len(found_imgs)
                window["-IMAGES-"].update()

            else:
                window["-TOUT-"].update(f"Founded images {len(found_imgs)} with query - {query_text}")

                images = get_images_grid("./_SEARCHRESULTS/")
                num_images = min(len(images), FIX_SHOW_IMAGES)
                num_columns = min(num_images, 3)
                num_rows = (num_images + num_columns - 1) // num_columns

                image_layout = []
                #try:
                for i, image_file in enumerate(images[:FIX_SHOW_IMAGES]):

                    img_path = os.path.join(folder, image_file)

                    temp_image_path, image_element = output_imgs(img_path, i)
                    image_layout.append(image_element)


                    window["-IMAGES-"].update(filename=temp_image_path)
                    #window["-IMAGES-"].update(filename=temp_image_path)
                    os.remove(temp_image_path)
                #except:
                #print(f"Error loading image: {img_path}")

                # Reshape image layout into rows and columns
                #print(len(image_layout))
                #image_rows = [image_layout[i:i+num_columns] for i in range(0, len(image_layout), num_columns)]
                #print(image_rows[0][0].size)
                # Recreate column with the new image elements


                # Update window layout with image grid
                #window["-IMAGES-"].update(data=image_rows)

    window.close()

nm()


