

import os
import sys
import pprint


WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(WORKING_DIR, '../'))


import torch
import cv2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.hub.load('app/neural_model/yolov5', 'custom', path='weight_test/best.pt', source='local', device=device)
model.eval()
names = model.module.names if hasattr(model, 'module') else model.names


image = cv2.imread('test/image_test/test_anh_duong.jpg')
results = model(image, size=int(640)).xyxy
output = []
for result in results:
    for r in result:
        output.append({'xmin': int(r[0]), 
                        'ymin': int(r[1]), 
                        'xmax': int(r[2]), 
                        'ymax': int(r[3]), 
                        'confidence': round(float(r[4]), 3), 
                        'polygons': [[]], 
                        'label': names[int(r[5])],
                        'label_index': int(r[5])
                        })
pprint.pprint(output)