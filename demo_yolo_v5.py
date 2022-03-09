from sahi.model import Yolov5DetectionModel,ScaleYoloV4Model
from sahi.predict import get_prediction, get_sliced_prediction, predict
import cv2

# set image
img = './https:__s3.amazonaws.com_mc-ai_dataset_india_20190312_1_IMG_20170417_101024.jpg'
image = cv2.imread(img)
h,w,_ = image.shape
#print(image.shape)

# detection_model = Yolov5DetectionModel(
#     model_path='yolov5s.pt',
#     confidence_threshold=0.1,
#     device="cuda:0", # or 'cuda:0'
# )

detection_model_1 = ScaleYoloV4Model(
    model_path='scratch.pt',
    confidence_threshold=0.5,
    device='cuda:0'
)

#result = get_prediction(img, detection_model_1)
#result.export_visuals(export_dir="demo_data/")

result = get_sliced_prediction(
    img,
    detection_model_1,
    slice_height = h/2,
    slice_width = w/2,
    overlap_height_ratio = 0.5,
    overlap_width_ratio = 0.5
)

print(result.to_coco_predictions())
result.export_visuals(export_dir="demo_data1/")







# perform inference
#results = model(img)

# inference with larger input size
#results = model(img, size=1280)

# inference with test time augmentation
#results = model(img, augment=True)
#print(results.xyxy)
# parse results
