from detectron2.config import get_cfg
from predictor import VisualizationDemo
import cv2
import time

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

if __name__ == "__main__":
    path = r'/home/ps/kp_deployment/kp_deployment.yaml'
    image = r'/home/ps/kp_deployment/23_4.jpg'
    output = r'/home/ps/kp_deployment/output_test5.png'

    cfg = setup_cfg(path)
    demo = VisualizationDemo(cfg)

    t0 = time.time()
    frame = cv2.imread(image)
    predictions, visualized_output = demo.run_on_image(frame)
    print('watest time %5f'%(time.time()-t0))
    print((predictions['instances'].pred_keypoints[0,3,0]))
    cv2.imwrite(output,visualized_output.get_image()[:, :, ::-1])