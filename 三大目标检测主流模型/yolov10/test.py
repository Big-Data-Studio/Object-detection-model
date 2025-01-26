
from ultralytics import YOLOv10 

#def yolov10_inference(image, model_path, image_size, conf_threshold):
#   model = YOLOv10(model_path)
    
#    model.predict(source=image, imgsz=image_size, conf=conf_threshold, save=True)
    
#    return model.predictor.plotted_img[:, :, ::-1]

model = YOLOv10("runs/detect/train2/weights/best.pt")
results = model.predict(source="C:/Users/86159/Desktop/1000", imgsz=640, conf=0.05, save=True)
pass
