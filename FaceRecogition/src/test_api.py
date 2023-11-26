import requests
from datetime import datetime

# API endpoint
update_door_state_url = ""
post_human_image_url = ""

# Function to update door state
def update_door_state(state):
    payload = {"state": state}
    requests.put(update_door_state_url, json=payload)

# Function to post human image
def post_human_image(name, image_base64, state):
    current_time = datetime.now().strftime("%H:%M:%S %d/%m/%Y")
    payload = {"name": name , "image": image_base64, "dateTime": current_time, "is_homie": state} 
    requests.post(post_human_image_url, json=payload)
    