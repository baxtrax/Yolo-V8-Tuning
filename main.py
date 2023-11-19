from ultralytics import YOLO

# Reference for yolov8 custom training:
# https://learnopencv.com/train-yolov8-on-custom-dataset/

model = YOLO('yolov8n.pt')
model.train(data='datasets/drone_real.yaml', name='drone_real_train', epochs=1, batch=8)
model.val(data='datasets/drone_real.yaml', name='drone_real_val', batch=1)
# model.track(source='cars_on_highway (1080p).mp4', show=True, conf=0.5)