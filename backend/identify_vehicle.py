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


    # get vehicle data
    def identify_vechile_data( self , image_path ) -> list:

        # this returns results of the prediction 
        results = self.vechile_identification_model(image_path)

        # empty list for identified vechiles
        vehicle_data = []

        # loop through the results
        for result in results:
            for box in result.boxes:

                # this variable stores the vechile type
                vehcile_type = result.names[int(box.cls[0])]

                vehicle_data.append({
                    'vehcile_type' : vehcile_type ,
                    'confidence' : float(box.conf[0]) , 
                    'coordinates' : box.xyxy[0].int().tolist() ,
                    'image_path' : image_path
                })
                
        # return the data to the function 
        return vehicle_data

# ------------------------------------------------------------------------------

    # fucntion for getting the vechile number if exits
    def get_vechile_license_plate_number(self , image_path ) -> Any:

        # get the coords using the identify_vechile_type function
        vehicle_data = self.identify_vechile_data( image_path )

        # process the image using cv2 
        processed_image = cv2.imread(image_path)

        # loop into the items 
        for vehicle in vehicle_data:

            # get the coords of the vechile detected
            vehile_x1 , vehile_y1 , vehile_x2 , vehile_y2 = vehicle['coordinates']

            # crop the image using cv2 [vehicle]
            vehile_cropped_image = processed_image[ vehile_y1:vehile_y2 , vehile_x1:vehile_x2 ]

            # check if the cropped image has license plate visible in it 

            if()


# ------------------------------------------------------------------------------



c  = VehicleInfo()

print(c.identify_vechile_data(r'img\number_plated_car.jpg'))