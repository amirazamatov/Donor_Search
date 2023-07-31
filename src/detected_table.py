import os
import torch
import torchvision
from torchvision import transforms
from torch import nn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.ops.boxes import nms

import numpy as np
import math
import cv2
import matplotlib.pyplot as plt

def imshow(inp,  title=None, plt_ax=plt, default=False, mode=None, color = None):
    if mode=='tensor':
        inp = inp.numpy().transpose((1, 2, 0))
        inp = np.clip(inp, 0, 1)
        plt_ax.imshow(inp, 'gray')
    else:
        plt_ax.imshow(inp, cmap=color)
    try:
        if title is not None:
            plt_ax.set_title(title)
    except:
        plt.title(title)
    plt_ax.grid(False)

def create_model(num_classes, pretrained=False):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def draw_predict_real_docs(file, iou_threshold=0., threshold=0.8, scale_percent=100):
    model = create_model(2)
    model.load_state_dict(torch.load('fasterrcnn_resnet50_fpn_v0_step_2.pth', map_location=torch.device('cpu')))

    img = cv2.imread(str(file))
    crop_img = img[int(img.shape[0] / 3):, ]
    crop_img_ = crop_img / 255.
    crop_img_ = torch.from_numpy(crop_img_).permute(2, 0, 1).unsqueeze(0).to(torch.float)

    model.eval()
    predict = model(crop_img_)
    ind = nms(predict[0]['boxes'], predict[0]['scores'], iou_threshold).detach().numpy()
    for i, box in enumerate(predict[0]['boxes'][ind]):
        if predict[0]['scores'][i] > threshold:
            crop_image = crop_img.copy()
            # нарисуем бокс
            cv2.rectangle(crop_img,
                          (int(box[0]), int(box[1])),
                          (int(box[2]), int(box[3])),
                          (255, 0, 0), 5)

            # границы обрезки по контурам бокса + небольшой запас по границам бокса
            border_up, border_down = int(crop_image.shape[0] * 0.02), int(crop_image.shape[0] * 0.055)
            carplate_image = crop_image[int(box[1]) - border_up:int(box[3]) + border_down, :]

    width = int(crop_img.shape[1] * scale_percent / 100)
    height = int(crop_img.shape[0] * scale_percent / 100)
    dim = (width, height)
    crop_img = cv2.resize(crop_img, dim)

    plt.suptitle(f'Detected table')
    imshow(crop_img)
    plt.show()

    return carplate_image


def preprocessing_image(img):
    koef = 3500 / img.shape[0]
    img = cv2.resize(img, (int(img.shape[1] * koef), int(img.shape[0] * koef)), cv2.INTER_LINEAR)

    # контраст
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(1, 1))
    cl = clahe.apply(l_channel)
    limg = cv2.merge((cl, a, b))
    img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # резкость
    g_img = cv2.GaussianBlur(img, (0, 0), 3)
    img = cv2.addWeighted(img, 5, g_img, -3.2, 0)
    for i in range(3):
        img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR, dstCn=3)

    return img


def longest_line(image):
    image = cv2.resize(image, (int((1000)),
                               int((image.shape[0] * 1000 / image.shape[1]))),
                               interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    inverted_image = cv2.bitwise_not(blur)

    # ищем горизонтальные линии:
    kernel_hor = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    hor = np.array([[1, 1, 1, 1, 1, 1]])
    vertical_lines_eroded_image = cv2.erode(inverted_image, hor, iterations=5)

    # расширяю линии
    vertical_lines_eroded_image = cv2.dilate(vertical_lines_eroded_image, hor, iterations=5)

    # расширяю все
    vertical_lines_eroded_image = cv2.dilate(vertical_lines_eroded_image, kernel_hor, iterations=3)
    vertical_lines_eroded_image = cv2.threshold(vertical_lines_eroded_image, 110, 255, cv2.THRESH_BINARY)[1]

    # Обнаружение границ в изображении с помощью оператора Canny.
    edges = cv2.Canny(vertical_lines_eroded_image, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=105, minLineLength=100, maxLineGap=5)
    longest_line = max(lines, key=lambda line: line[0][2] - line[0][0])

    return longest_line

def find_angle(table_line):
    PI = 3.14159265
    angle_rad = math.atan2(table_line[0][3] - table_line[0][1], table_line[0][2] - table_line[0][0])
    angle_deg = math.degrees(angle_rad)
    return angle_deg

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def prepare_image(image):
    line = longest_line(image.copy())
    angle = find_angle(line)
    image = rotate_image(image, angle)

    # Преобразование изображения в оттенки серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # инвертируем цвета, фон становится черным
    inverted_image = cv2.bitwise_not(gray)

    #plt.suptitle(f'Carved table')
    #imshow(inverted_image, color='gray')

    return inverted_image

def extract_image(file):
    img = cv2.imread(file)
    plt.suptitle(f'Предварительный просмотр')
    imshow(img)
    plt.show()
    # детекция таблицы и обрезка
    crop_image = draw_predict_real_docs(file)
    crop_image = preprocessing_image(crop_image)
    crop_image = prepare_image(crop_image)

    return crop_image