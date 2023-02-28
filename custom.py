import torch
model = torch.hub.load("ultralytics/yolov5", "yolov5s") 
img = "https://ultralytics.com/images/zidane.jpg"  
results = model(img)
labels, cord_thres = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()
print(labels)
print(cord_thres[0][2]-cord_thres[0][0])