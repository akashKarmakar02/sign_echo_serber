from io import BytesIO
from typing import List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from PIL import Image

from util import detect_hand_gesture

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

connected_clients: List[WebSocket] = []


@app.websocket('/predict')
async def predict(websocket: WebSocket):
    await websocket.accept()
    print(f"{websocket} connected")
    index = 0
    connected_clients.append(websocket)

    try:
        while True:
            data = await websocket.receive_bytes()

            image = Image.open(BytesIO(data))

            data = detect_hand_gesture(image)
            print(data)

            # Example: Echo back the data
            for client in connected_clients:
                try:
                    await client.send_json(data)
                    print(index)
                    index += 1
                except Exception as e:
                    print(f"Error sending data to client: {e}")
                    connected_clients.remove(client)
    except WebSocketDisconnect as e:
        print(f"Client disconnected: {e}")
        connected_clients.remove(websocket)
        index = 0

# uvicorn.run(app=app, port=8000, host="0.0.0.0")
