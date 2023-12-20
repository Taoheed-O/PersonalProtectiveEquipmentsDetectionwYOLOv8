from ultralytics import YOLO

# Load the model.
model = YOLO('yolov8n.pt')
 
# Training.
results = model.train(
   data='data.yaml',
   imgsz=640,
   epochs=10,
   batch=8,
)

# # predict
# model.predict(source='cars_humans_bikes.jpg', save=True, conf=0.5, save_txt=True)