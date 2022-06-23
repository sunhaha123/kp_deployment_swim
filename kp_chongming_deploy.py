from detectron2.config import get_cfg
from predictor import VisualizationDemo
import cv2
import time
import json
import numpy as np

from flask import Flask,request
app = Flask(__name__)

def setup_cfg(config_file):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
    # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
    # add_panoptic_deeplab_config(cfg)
    cfg.merge_from_file(config_file)
    # cfg.merge_from_list(config_file)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.5
    cfg.freeze()
    return cfg


@app.route("/body_analysis",methods=["POST"])
def  dataPoints():
    if request.method=='POST':
        t0 = time.time()
        img_bytes = request.files['image'].read()
        alarm_img = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(alarm_img, 1)
        y, x = frame.shape[0:2]
        frame = cv2.resize(frame, (int(x * 3), int(y * 3)))
        try:
            predictions, visualized_output = demo.run_on_image(frame)
            print('reco_point_ watest time %5f' % (time.time() - t0))
            k = {"left_shoulder_x" : predictions['instances'].pred_keypoints[0,1,0].item(),
            "left_shoulder_y"  : predictions['instances'].pred_keypoints[0,1,1].item(),
            "left_hip_x" : predictions['instances'].pred_keypoints[0,3,0].item(),
            "left_hip_y" : predictions['instances'].pred_keypoints[0, 3, 1].item(),
            "right_shoulder_x" : predictions['instances'].pred_keypoints[0,4,0].item(),
            "right_shoulder_y" : predictions['instances'].pred_keypoints[0, 4, 1].item(),
            "right_hip_x" : predictions['instances'].pred_keypoints[0, 6, 0].item(),
            "right_hip_y" : predictions['instances'].pred_keypoints[0, 6, 1].item()}
            finally_json = json.dumps(k, ensure_ascii=False, ).replace("'", "\"")

            if time.time()%10 ==0:
                cv2.line(frame, (int(k['left_shoulder_x']), int(k['left_shoulder_y'])),
                         (int(k['left_hip_x']), int(k['left_hip_y'])), (255, 0, 255), 2)
                cv2.line(frame, (int(k['right_shoulder_x']), int(k['right_shoulder_y'])),
                         (int(k['right_hip_x']), int(k['right_hip_y'])), (255, 0, 255), 2)
                cv2.imwrite('/home/ps/kp_deployment/0925/%s.jpg'%time.time(), frame)

            return finally_json
        except:
            finally_json = json.dumps('', ensure_ascii=False, )
            return finally_json





if __name__ == "__main__":
    path = r'/home/ps/kp_deployment/kp_deployment.yaml'
    cfg = setup_cfg(path)
    demo = VisualizationDemo(cfg)
    app.run(host='0.0.0.0')