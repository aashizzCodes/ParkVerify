# this file manages the location 



from typing import Any


class CheckLocation:

    def __init__(self) -> None:
        
        self.defined_location_list = {} 


    def get_parking_status(self , location ) -> dict:
        
        # loop through the self.defined_location_list return True if illegal_parking else False
        if location in  self.defined_location_list:
            return { " illegal_parking_status " : True}
        
        else:
            return { " illegal_parking_status " : False}