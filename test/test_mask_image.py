import os
import sys

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(WORKING_DIR, '../'))

import cv2
import numpy as np

def read_image_mask():
    # image = cv2.imread('test/image_test/mask.png', cv2.IMREAD_UNCHANGED)
    # print(image.shape)
    # print(image[1][1])
    # print(image[48][45])
    # print(image[50][46])
    # print(image[81][74])
    # bgr = image[:, :, :3]
    # alpha = image[:, :, 3]
    # cv2.imshow("BGR Image", bgr)
    # cv2.waitKey(0)

    # cv2.imshow("Alpha Channel", alpha)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    image = cv2.imread('test/image_test/mask.png', cv2.IMREAD_GRAYSCALE)
    height, width = image.shape
    mask = np.zeros((height, width), dtype=bool)

    mask[np.where(image>0)] = True

    in_image_point = np.transpose(np.where(mask))

    for point in in_image_point:
        print(f"Pixel coordinates: ({point[0]}, {point[1]})")


def create_image(url_image):
    image = cv2.imread(url_image, cv2.IMREAD_COLOR)
    # image = cv2.resize(image, (113, 200))
    print(image[0][0])
    ratio = image.shape[0]/200
    arr = (34, 47, 68, 81)
    
    (xmin, ymin, xmax, ymax) = np.array(tuple(element * ratio for element in arr), dtype=np.int64)
    print(xmin, xmax, ymin, ymax)
    img_brightness = np.zeros(image.shape[:2], dtype=np.uint8)
    img_brightness[ymin:ymax, xmin:xmax] = 255
    fg = cv2.bitwise_or(image, image, mask=img_brightness)
    image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)

    alpha = 0.55
    blended_image = cv2.addWeighted(image, alpha, fg, 1-alpha, 0, image)
    print(image[0][0])

    cv2.imshow('img1', blended_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    url_image = 'test/image_test/vu.jpg'
    create_image(url_image)