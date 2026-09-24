import sys
import argparse
import os


from ultralytics import YOLO

def main(opt):
    weights = opt.weights

    model = YOLO(weights)

    model.info()
    
    results = model.val(data='coco128.yaml', 
                    imgsz=640,  
                    batch=2,
                    )

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default= r'yolov8n.pt', help='initial weights path')
    parser.add_argument('--artifact_alias', type=str, default='latest', help='W&B: Version of dataset artifact to use')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)