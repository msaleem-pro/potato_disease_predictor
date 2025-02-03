from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
import numpy as np
import tensorflow as tf
import gdown

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
model_url = "https://drive.google.com/uc?id=1T4091isCyD4xdWYrG4Oesvx_3wPvcbS3"
output = "potato_disease_model.h5"

# Download model if not exists
import os
if not os.path.exists(output):
    gdown.download(model_url, output, quiet=False)

model = tf.keras.models.load_model(output)
classes = ["Early Blight", "Healthy", "Late Blight"]
@app.get("/")
def welcome():
    return {"msg": "Welcome to Potato Disease Predcitor"}

@app.post("/upload/")
async def upload_and_segment_image(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        return {"error": "File is not an image."}

    image = Image.open(file.file)
    img = image.resize((256, 256))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    img_array = img_array.astype('float32') / 255.
    pre = model.predict(img_array)
    predicted_class = classes[np.argmax(pre)] 
    confidence = float(np.max(pre))*100
    early = float(pre[0,0])
    healthy = float(pre[0,1])
    late = float(pre[0,2])
    


    return {
        "status":True,
        "prediction": predicted_class,
        "confidence": confidence,
        "all": {"Early Blight": f"{early:.2f}", "Healthy": f"{healthy:.2f}","Late Blight": f"{late:.2f}"},
        
        
        
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
