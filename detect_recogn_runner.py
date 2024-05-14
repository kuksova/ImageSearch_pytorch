import os.path

from detection_textboxes_my.main_detection import run_detector_craft, crop_resize
from crnn_recognition_my.run_sequence_prediction import run_one_sample
from detection_textboxes_my.detection_text_model import get_detector

from crnn_recognition_my.predict_text import predict_text

import numpy as np
from PIL import Image
import time
import cv2

#class
STATISTICS = '.\_STATISTICS'


def main(img_path): # batch of images
    image = Image.open(img_path).convert('L')
    image = image.resize((608, 1280)) # w, h
    image = np.array(image)# h, w

    result = run_detector_craft(img_path)[0] # Model CRAFT size: 79.23303985595703 MB;  3.15 sec

    res = []
    i = 0
    bboxes = []
    for bbox in result:
        #(tl, tr, br, bl) = bbox

        tl = (int(bbox[0]), int(bbox[1]))
        br = (int(bbox[4]), int(bbox[5]))

        box_h = br[1] - tl[1]
        box_w = br[0] - tl[0]
        #print(i, box_h, box_w)

        cropped_image = image[tl[1]:br[1], tl[0]:br[0]]
        #cropped_image = Image.fromarray(cropped_image, 'L')
        crop_img = crop_resize(cropped_image)  # transform to (32, 100)
        bboxes.append(crop_img)

        #cv2.rectangle(image, tl, br, (0, 255, 0), 2)

        #cv2.imwrite(str(i) + 'cropped_image16.jpg', np.array(crop_img))

        # Skip the square box images , working on with boxes only box_w>>box_h
        # found out this is not a good idea, I've lost some useful boxes
        """
        if box_h > 13 and box_w > 50:

            cropped_image = image[tl[1]:br[1], tl[0]:br[0]]
            # cropped_image = image[tl[1]:br[1], min(tl[0], br[0]):max(tl[0], br[0])]

            crop_img = crop_resize(cropped_image) # transform to (32, 100)

            bboxes.append(crop_img)

            cv2.rectangle(image, tl, br, (0, 255, 0), 2)

            #cv2.imwrite(str(i) + 'cropped_image16.jpg', np.array(crop_img))


        """

        #pred_srt, confidence_score = run_one_sample(crop_img)
        #if confidence_score > 0.7:
        #    res.append([pred_srt, confidence_score])

        #print(pred_srt, confidence_score)
        i = i+1
        #if i == 5:
        #    break

    #d, fn = os.path.split(img_path)
    #name, ex = os.path.splitext(fn)
    #img1_path = os.path.join('.', f"{name}_box{ex}")
    #cv2.imwrite( img1_path, np.array(image))

    # If the image has no text, skip it
    # Save an id of these images for manual checking
    txt_name = os.path.join(STATISTICS, "img_no_text.txt")
    if not bboxes: # if it's empty
        with open(txt_name, "w") as outfile:
            outfile.write(img_path)
        return []
    else:
        out_text = predict_text(bboxes) # Text extracted from 1 image
        return out_text

if __name__ == "__main__":
    img_path = '_IMAGESTORAGE/09_Sep/01a28453a0a988b3290d5f746c882578d66574d432.jpg'
    t0 = time.time()
    text_out = main(img_path)
    #get_detector('cpu')
    print("Recognition time per the whole image = ", time.time() - t0)
    print("Extracted text ", text_out)

# I run a recognition in for loop, how to check if pretrained model is already uploaded once and I don't need it to upload it again