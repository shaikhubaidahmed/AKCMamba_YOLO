import sys
import argparse
import os


from ultralytics import YOLO

'''
python train_v8.py --cfg 

'''

def main(opt):
    yaml = opt.cfg
    weights = opt.weights
    # model = YOLO(weights)
    model = YOLO(yaml)


    # print(model)

    model.info()

    results = model.train(data='coco128.yaml', 
                        epochs=20, 
                        imgsz=640, 
                        workers=0, 
                        batch=2,
                        )

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default= r'E:\Program Files\Anaconda\envs\pythonProject2\AKCMamba_YOLO\ultralytics\cfg\models\V8\YOLOv8.yaml', help='initial weights path')
    parser.add_argument('--weights', type=str, default='yolov8n.pt', help='')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)