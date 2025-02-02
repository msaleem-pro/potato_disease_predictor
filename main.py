from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps
from fastapi.staticfiles import StaticFiles
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

# Folders for uploaded and processed images
UPLOAD_FOLDER = "uploaded_images"
PROCESSED_FOLDER = "processed_images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Serve static files for both folders
app.mount("/uploads", StaticFiles(directory=UPLOAD_FOLDER), name="uploads")
app.mount("/processed", StaticFiles(directory=PROCESSED_FOLDER), name="processed")

@app.post("/upload/")
async def upload_and_segment_image(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        return {"error": "File is not an image."}

    # Save the original image
    file_location = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_location, "wb") as buffer:
        buffer.write(await file.read())

    # Open image for processing
    image = Image.open(file_location)

    # Convert to grayscale for segmentation
    gray_image = ImageOps.grayscale(image)

    # Apply simple threshold segmentation
    threshold = 128
    segmented_image = gray_image.point(lambda p: 255 if p > threshold else 0)

    # Save the segmented image
    processed_filename = f"segmented_{file.filename}"
    processed_location = os.path.join(PROCESSED_FOLDER, processed_filename)
    segmented_image.save(processed_location)

    # Return URLs instead of file paths
    return {
        "message": "Image uploaded and segmented successfully!",
        "original_image": f"uploads/{file.filename}",
        "segmented_image": f"processed/{processed_filename}"
    }
