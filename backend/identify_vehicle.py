'''
Docstring for backend.identify_vehicle

* this file identifies the vechile in the image using the yolo11n model
* returns the name of the vechile and identified vechile in the image and vechile number and respective confidence

'''



from ultralytics import YOLO
from typing import Any


# ------------------------------------------------------------------------------

class VehicleInfo:
    # inital function
    def __init__(self) -> None:
        # models path and initialization
        self.vechile_identification_model = YOLO(r"model\yolo11n.pt")
        self.license_plate_identification_model = YOLO(r'model\license-plate-finetune-v1n.pt')


    # identify the vechile by type 
    def identify_vechile_type( self , image_path ) -> Any:

        # this returns results of the prediction 
        results = self.vechile_identification_model(image_path)

        # empty array for identified vechiles
        vehicle_data = []

        # loop through the results
        for result in results:
            for box in result.boxes:

                # this variable stores the vechile type
                vehcile_type = result.names[int(box.cls[0])]

                vehicle_data.append({
                    'vehcile_type' : vehcile_type ,
                    'confidence' : float(box.conf[0]) , 
                    'coordinates' : box.xyxy[0].int().tolist()
                })
                
        # return the detials to the function 
        return vehicle_data

# ------------------------------------------------------------------------------

    # fucntion for getting the vechile number if exits
    def get_vechile_license_plate_number(self,image_path):
        
        # get image and check if the user plate exits return the values if True
        pass

# ------------------------------------------------------------------------------

    # function to get the number plate data 
    def get_plate_data(self , cropped_img_path) -> Any:

        pass

# ------------------------------------------------------------------------------
