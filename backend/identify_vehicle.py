'''
Docstring for backend.identify_vehicle

* this file identifies the vechile in the image using the yolo11n model
* returns the name of the vechile and identified vechile in the image and vechile number and respective confidence

'''



from ultralytics import YOLO
from typing import Any


class vehicle_info:
    # inital function
    def __init__(self) -> None:
        # model path
        self.Model = YOLO(r"model\yolo11n.pt")

    # identify the vechile by type 
    def identify_vechile_type( self , image_path ) -> Any:

        # this returns results of the prediction 
        results = self.Model(image_path)

        # loop through the 
        for result in results:
            for box in result.boxes:
                return { 'vechicle_Type' : result.names[int(box.cls)] }