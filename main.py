from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
import os

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure folders exist in serverless environments (Vercel may reset on each deploy)
UPLOAD_FOLDER = "/tmp/uploaded_images"
PROCESSED_FOLDER = "/tmp/processed_images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Serve static files (may not work perfectly on Vercel, consider using cloud storage like S3)
app.mount("/uploads", StaticFiles(directory=UPLOAD_FOLDER), name="uploads")
app.mount("/processed", StaticFiles(directory=PROCESSED_FOLDER), name="processed")


@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

@app.post("/upload/")
async def upload_and_segment_image(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        return {"error": "File is not an image."}

    file_location = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_location, "wb") as buffer:
        buffer.write(await file.read())

    image = Image.open(file_location)
    gray_image = ImageOps.grayscale(image)

    threshold = 128
    segmented_image = gray_image.point(lambda p: 255 if p > threshold else 0)

    processed_filename = f"segmented_{file.filename}"
    processed_location = os.path.join(PROCESSED_FOLDER, processed_filename)
    segmented_image.save(processed_location)

    return {
        "message": "Image uploaded and segmented successfully!",
        "original_image": f"/uploads/{file.filename}",
        "segmented_image": f"/processed/{processed_filename}"
    }

# This is important for Vercel to know how to run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
