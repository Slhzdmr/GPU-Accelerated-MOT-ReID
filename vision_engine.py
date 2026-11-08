import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from ultralytics import YOLO

class VisionEngine:
    def __init__(self,device):
        self.device = device
        self.detector = YOLO("yolov8n.pt")
        self.reid_model = self._get_reid_model()

    def _get_reid_model(self):
        model = models.resnet18(weights="DEFAULT")
        model.fc == nn.Identity() #we need the embeddings
        model = model.to(self.device)
        model.eval()
        return model

    def preprocess_crop(self,crop_img):
        crop_rgb = cv2.cvtColor(crop_img,cv2.COLOR_BGR2RGB)
        crop_resized = cv2.resize(crop_rgb,(128,256))
        tensor_img = torch.from_numpy(crop_resized).float() / 255.0
        tensor_img = tensor_img.permute(2,0,1).unsqueeze(0)
        return tensor_img

    def extract_features(self,frame):
        results = self.detector.track(frame,persist=True,tracker="bytetrack.yaml",classes=[0],verbose=False)

        if results[0].boxes.id is None:
            return [], None

        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy()

        valid_tensors = []
        valid_data = []

        for box, track_id in zip(boxes,track_ids):
            x1,y1,x2,y2 = map(int,box)
            h,w,_ = frame.shape
            x1,y1 = max(0,x1), max(0,y1)
            x2,y2 = min(w,x2), min(h,y2)

            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0 or person_crop.shape[0] < 10 or person_crop.shape[1] < 10:
                continue

            tensor_img = self.preprocess_crop(person_crop).to(self.device)
            valid_tensors.append(tensor_img)
            valid_data.append({"id": track_id, "box": (x1, y1, x2, y2)})

        if len(valid_tensors) == 0:
            return [], None

        batch_tensor = torch.cat(valid_tensors,dim=0)


        with torch.no_grad():
            raw_embeddings = self.reid_model(batch_tensor)
            normalized_embeddings = F.normalize(raw_embeddings,p=2, dim=1)#used p2 to make torch.mv calculate cos similarity

        return valid_data, normalized_embeddings







