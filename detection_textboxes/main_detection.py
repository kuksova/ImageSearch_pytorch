from PIL import Image
from torchvision import transforms
import torch
import time

from detection_textboxes_my.detection_text_model import test_net, get_detector

#import imgproc
import cv2
import numpy as np



device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_height = 32


def run_detector_craft(img_path):


    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((1280, 608)),  # h,w
        # transforms.Resize((32,100)), # h,w
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    #transformer = dataset.resizeNormalize((100, 32)) # height=32 but also width=100
    image = np.array(Image.open(img_path).convert('RGB'))

    # resize
    canvas_size = 1280
    mag_ratio = 1.5
    #img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, canvas_size,
    #                                                                      interpolation=cv2.INTER_LINEAR,
    #                                                                      mag_ratio=mag_ratio)
    ratio_h = ratio_w = 1  # / target_ratio

    image = transform(image)
    image = torch.unsqueeze(image, 0)

    craft_model = get_detector(device)
    print(device)
    print("Craft pretrained model uploaded.")
    t1 = time.time()

    result = test_net(craft_model, image, device, ratio_w, ratio_h)
    print("time get detection boxes ", time.time() - t1, ' sec ')

    return result


def crop_resize(cropped_image):

    ratio = 1.0
    try:
        ratio = cropped_image.shape[1] / cropped_image.shape[0]  # width / height
    except ZeroDivisionError:
        print("Error: Division by zero!")
    if ratio < 1.0:
        ratio = 1. / ratio
        crop_img = cv2.resize(cropped_image, (model_height, int(model_height * ratio)),
                              cv2.INTER_AREA)  # interpolation=Image.Resampling.LANCZOS)
    else:
        crop_img = cv2.resize(cropped_image, (int(model_height * ratio), model_height),
                              cv2.INTER_AREA)  # interpolation=Image.Resampling.LANCZOS)
    return crop_img


def get_boxes_parser(result, img_path):

    image = Image.open(img_path).convert('L')
    #img_cv_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image = image.resize((608, 1280))
    image = np.array(image)

    bboxes = []
    for bbox in result[0]:

        # (tl, tr, br, bl) = bbox
        tl = (int(bbox[0]), int(bbox[1]))
        tr = (int(bbox[2]), int(bbox[3]))
        br = (int(bbox[4]), int(bbox[5]))
        bl = (int(bbox[6]), int(bbox[7]))

        bboxes.append(bbox)

        cropped_image = image[tl[1]:br[1], tl[0]:br[0]]
        crop_img = crop_resize(cropped_image)


        print(crop_img.shape)
        break
        #print(cropped_image.shape)

        #cv2.rectangle(image, tl, br, (0, 255, 0), 2)
    return bboxes







if __name__ == "__main__":
    img_path = '../Sveta_imgs/IMG_0026.PNG'

    result = run_detector_craft(img_path)
    bboxes = get_boxes_parser(result, img_path)









