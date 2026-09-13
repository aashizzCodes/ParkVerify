"""
Docstring for backend.identify_vehicle

* this file identifies the vechile in the image using the yolo11n model
* returns the name of the vechile and identified vechile in the image and vechile number and respective confidence

"""

from ultralytics import YOLO
from typing import Any
import cv2
import numpy
import easyocr

# ------------------------------------------------------------------------------


class VehicleInfo:

    # inital function
    def __init__(self) -> None:
        # models path and initialization
        self.vehicle_model = YOLO(r"model\yolo11n.pt")
        self.license_plate_model = YOLO(r"model\license-plate-finetune-v1n.pt")
        self.ocr_reader = easyocr.Reader(['en'])


    # Public Functions
    # identify function
    def identify(self , image:str) -> list:

        # get raw data and store it
        results = self._get_raw_vehicle_data(image)

        # crop the image into the detected coordinates
        for vehicle in results:
            # crop the image using the coordinates
            cropped_image = self._crop_image(vehicle['coordinates'] , image)

            # call the license plate function to get the license plate data
            license_plate_data = self._get_license_plate_data(cropped_image)
            
            # add the license plate data to the vehicle data 
            vehicle.update(license_plate_data)
        
        # return the results
        return results



    # Helper Functions

    # get raw vehicle detection data
    def _get_raw_vehicle_data(self , image:str ) -> list:

        # empty list to store the vehicle data
        vehicle_data = []

        # using the vehicle model to get the information
        results = self.vehicle_model(image)

        # loop into the results
        for result in results:
            for box in result.boxes:
                vehicle_data.append(
                    {
                        "class": result.names[int(box.cls)],
                        "coordinates": box.xyxy[0].int().tolist(),
                        "confidence": float(box.conf[0]),
                        "source_path": image,
                    }
                )

        # return the vehicle data
        return vehicle_data


    # crop image function
    def _crop_image(self, coordinates: list , image:numpy.ndarray) -> numpy.ndarray:

        # set the coordinates here
        x1, y1, x2, y2 = coordinates
        # process the image
        processed_image = cv2.imread(image)
        # crop the image using the cv2
        cropped_image = processed_image[y1:y2, x1:x2]
        # return the image  
        return cropped_image


    # get license plate data function
    def _get_license_plate_data(self, cropped_image: numpy.ndarray) -> dict:

        # empty dict to store the license plate data
        license_plate_data = {}

        # using the license plate model to get the information
        results = self.license_plate_model(cropped_image)
        
        # loop into the results
        for result in results:
            for box in result.boxes:
                    license_plate_data.update(
                        {
                            "plate_coordinates" : box.xyxy[0].int().tolist(),
                            "confidence" : float(box.conf[0]),
                        }
                    )
        
        # if no license plate is detected, return an empty dict
        if not license_plate_data:
            return {}
        
        # crop the license plate image using the coordinates
        license_plate_cropped_image = self._crop_image(license_plate_data['plate_coordinates'], cropped_image)

        # get the license plate number using the license plate cropped image
        license_plate_number =  self.ocr_reader.readtext(license_plate_cropped_image)

        # add the license plate number to the license plate data
        license_plate_data.update(
            {
                "license_plate_number" : license_plate_number[0][1] if len(license_plate_number) > 0 else None ,
                "license_plate_confidence" : license_plate_number[0][2] if len(license_plate_number) > 0 else None
            }
        )
        # return the license plate data
        return license_plate_data