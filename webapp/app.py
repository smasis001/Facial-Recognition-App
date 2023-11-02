"""Fast API Front End & Back Ends"""
# pylint: disable=E1101,C0413,W0718
from typing import List, Optional, Dict
import os
import sys
import base64
import cv2
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from starlette.templating import _TemplateResponse
from pydantic import BaseModel
import uvicorn
sys.path.append("../")
par_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, par_dir)
from facial_recognition import FaceRepresentation, FaceIdentification

# Initialize FastAPI and Facial Recognition Classes
app = FastAPI()
face_descriptor = FaceRepresentation()
face_identifier = FaceIdentification()

# CORS breaker
origins = ['*']
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Select directory for website templates
templates = Jinja2Templates(directory="webapp/templates")

# Mount static directory for JS and CSS files
app.mount("/static", StaticFiles(directory="webapp/static"), name="static")


class VerifyPacket(BaseModel):
    """Model for Verfification Data"""
    frame_enc:str
    bb:List[int]
    landmarks:List[List[int]]
    name:Optional[str] = None

@app.get("/", response_class=HTMLResponse)
async def main(
        request: Request
    ) -> _TemplateResponse:
    """Main function that returns a TemplateResponse for the main frontend page.

    Args:
        request (Request): The request object.

    Returns:
        _TemplateResponse: The TemplateResponse object.
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/identify/")
async def identify(
        packet:VerifyPacket = None
    ) -> Dict:
    """Identify a face in a given frame.

    Args:
        packet (VerifyPacket): The packet containing the frame and other information.
                               Defaults to None.

    Returns:
        dict: A dictionary containing the identification results. The dictionary has the following
              keys:
            - "name" (str): The name of the identified person. If no match is found, it is set
              to "nomatch".
            - "distance" (float): The distance between the identified person and the input frame.
            - "displayName" (str): A formatted string containing the name and distance.

    Raises:
        Exception: If there is an error uploading the frame.
    """
    if VerifyPacket is None:
        raise ValueError("Can't send an empty packet")
    try:
        frame_bin = base64.b64decode(packet.frame_enc.replace("data:image/png;base64,",""))
    except Exception as exc:
        return {"error": f"There was an error uploading the frame {exc}"}
    frame_np = np.frombuffer(frame_bin, np.uint8)
    frame_np = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)
    frame_rgb = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
    landmarks = [face_descriptor.convert_landmarks(packet.bb, packet.landmarks)]
    descriptors = face_descriptor.represent(frame_rgb, landmarks)
    if packet.name is not None:
        name_cnt = face_identifier.count(packet.name)
        if name_cnt < 10:
            face_descriptor.add_to_vectordb(np.array(descriptors),\
                                            ids=[packet.name])
            return {
                "message":f"Added `{packet.name}` to the Vector DB"
            }
        else:
            packet.name = None

    best_distances, best_neighbors = face_identifier.identify(descriptors, k=1)
    for neighbors, distances in zip(best_neighbors, best_distances):
        return {
                "name": neighbors[0],
                "distance": round(distances[0]*10000)/10000,
                "displayName": f"{neighbors[0]} {distances[0]:.3f}",
                "nameProvided": packet.name
            }
    return {
        "name":"nomatch",
        "distance":300,
        "displayName": "NO MATCH"
    }

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=80)
