'''
Docstring for backend.identify_vehicle

* this file identifies the vechile in the image using the yolo11n model
* returns the name of the vechile and identified vechile in the image and vechile number and respective confidence

'''



from ultralytics import YOLO
from typing import Any
import cv2


# ------------------------------------------------------------------------------

class VehicleInfo:
    # inital function
    def __init__(self) -> None:
        # models path and initialization
        self.vechile_identification_model = YOLO(r"model\yolo11n.pt")
        self.license_plate_identification_model = YOLO(r'model\license-plate-finetune-v1n.pt')

    