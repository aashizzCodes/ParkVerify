'''
Docstring for backend.identify_vehicle

* this file identifies the vechile in the image using the yolo11n model
* returns the name of the vechile and identified vechile in the image and vechile number and respective confidence

'''



from ultralytics import YOLO
from typing import Any


class VehicleInfo:
    # inital function
    def __init__(self) -> None:
        # model path
        self.Model = YOLO(r"model\yolo11n.pt")

    # identify the vechile by type 
    def identify_vechile_type( self , image_path ) -> Any:

        # this returns results of the prediction 
        results = self.Model(image_path)

        # empty array for identified vechiles
        vehicles = []

        # loop through the results
        for result in results:
            for box in result.boxes:

                # this variable stores the vechile type
                vehcile_type = result.names[int(box.cls[0])]

                vehicles.append({
                    'vehcile_type' : vehcile_type ,
                    'confidence' : float(box.conf[0]) , 
                    'coordinates' : box.xyxy[0].int().tolist()
                })
            
    # fucntion for getting the vechile number if exits
    def get_vechile_license_plate_number(self,image_path):
        pass