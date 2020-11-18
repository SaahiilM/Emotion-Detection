import os
import json #to work with the Json data
import requests #HTTP library 
import matplotlib #plotting library
import matplotlib.pyplot as plt #plotting library 
from matplotlib.patches import Rectangle #plotting library
from PIL import Image #Image library for python
from io import BytesIO 
import urllib.request #for opening of urls
from msrest.authentication import CognitiveServicesCredentials #to authenticate the access
from azure.cognitiveservices.vision.face import FaceClient #API for face detection

#creating an authenticated face client for the api
def get_face_client():
    """Create an authenticated FaceClient."""
    SUBSCRIPTION_KEY = "10be51f5bccf41a5b1a781f6bfd3ff0c"
    ENDPOINT = "https://demo-service.cognitiveservices.azure.com/"
    credential = CognitiveServicesCredentials(SUBSCRIPTION_KEY)
    return FaceClient(ENDPOINT, credential)
face_client = get_face_client()

url = input("\nEnter the url:\n")

response = requests.get(url)
image = Image.open(BytesIO(response.content))

attributes = ["emotion"]
include_id = True
include_landmarks = False

detected_faces = face_client.face.detect_with_url(url, include_id, include_landmarks, attributes, raw=True) #passsing the parameters to the api and the attributes that require in return.

if not detected_faces:
    raise Exception('No face detected in image')

#getting values of each emotion key
responseJson = detected_faces.response.json() 
emotionId = responseJson[0]["faceAttributes"]["emotion"]
angerId = responseJson[0]["faceAttributes"]["emotion"]["anger"]
contemptId = responseJson[0]["faceAttributes"]["emotion"]["contempt"]
disgustId = responseJson[0]["faceAttributes"]["emotion"]["disgust"]
fearId = responseJson[0]["faceAttributes"]["emotion"]["fear"]
happinessId = responseJson[0]["faceAttributes"]["emotion"]["happiness"]
neutralId = responseJson[0]["faceAttributes"]["emotion"]["neutral"]
sadnessId = responseJson[0]["faceAttributes"]["emotion"]["sadness"]
surpriseId = responseJson[0]["faceAttributes"]["emotion"]["surprise"]

#store values of emotion key
emotionSelect = [] 

emotionSelect.append(angerId)
emotionSelect.append(contemptId)
emotionSelect.append(disgustId)
emotionSelect.append(fearId)
emotionSelect.append(happinessId)
emotionSelect.append(neutralId)
emotionSelect.append(sadnessId)
emotionSelect.append(surpriseId)

#return the ID of emotion with max value
maxemotion = emotionSelect.index(max(emotionSelect))

#convert the ID to the associated emotion
def numbers_to_emotions(argument):
    switcher = {
        0: "angry",
        1: "contempt",
        2: "disgusted",
        3: "scared",
        4: "happy",
        5: "neutral",
        6: "sad",
        7: "surprised"
    }

    return switcher.get(argument, "nothing")

#display the emotion values from the list
x = responseJson[0]["faceAttributes"]["emotion"]
print("\nFollowing are the detected emotions and its confidence values:\n")
for k, v in x.items():
    print(k, v)


print("\nThe maximum confidence value detected is for the following emotion:", numbers_to_emotions(maxemotion))

#create bounding box for face
identifiedEmotion = numbers_to_emotions(maxemotion)
bboxTop = responseJson[0]["faceRectangle"]["top"]
bboxLeft = responseJson[0]["faceRectangle"]["left"]
bboxWidth = responseJson[0]["faceRectangle"]["width"]
bboxHeight = responseJson[0]["faceRectangle"]["height"]

bbox = []

bbox.append(bboxTop)
bbox.append(bboxLeft)
bbox.append(bboxWidth)
bbox.append(bboxHeight)

#plot image and annotations
topLabelPosx = (bboxLeft)
topLabelPosy = (bboxTop)
bottomLabelPosx = (bboxLeft)
bottomLabelPosy = (bboxTop+bboxHeight)

plt.imshow(image)
ax = plt.imshow(image, alpha=0.5)

origin = (bbox[1], bbox[0])
patch = Rectangle(origin, bbox[2], bbox[3], fill=False, linewidth=2, color='r')
ax.axes.add_patch(patch)




plt.annotate(identifiedEmotion, #text for label
             (bottomLabelPosx, bottomLabelPosy), #point to label
             textcoords="offset points", #offset for text
             xytext=(0, 0), #distance from text to points
             ha='left', #horizontal alignment 
             va='bottom',
             color="white",
             weight="bold",
             backgroundcolor="black"
             )

plt.axis("off")
plt.show()