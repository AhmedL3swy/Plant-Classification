import datetime
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI()
#use Static Files
app.mount("/images", StaticFiles(directory="images"), name="images")
model = load_model('InceptionV3_1.h5')

Classes = {
    0: 'Pepper__bell___Bacterial_spot',
    1: 'Pepper__bell___healthy',
    2: 'Potato___Early_blight',
    3: 'Potato___Late_blight',
    4: 'Potato___healthy',
    5: 'Tomato_Bacterial_spot',
    6: 'Tomato_Early_blight',
    7: 'Tomato_Late_blight',
    8: 'Tomato_Leaf_Mold',
    9: 'Tomato_Septoria_leaf_spot',
    10: 'Tomato_Spider_mites_Two_spotted_spider_mite',
    11: 'Tomato__Target_Spot',
    12: 'Tomato__Tomato_YellowLeaf__Curl_Virus',
    13: 'Tomato__Tomato_mosaic_virus',
    14: 'Tomato_healthy'
}

def getPrediction(image):
    # Convert Image to Array
    img_array = img_to_array(image)
    # Expand Image Array
    img_array = tf.expand_dims(img_array, 0)
    # Predict Image
    predictions = model.predict(img_array)
    # get the class with the highest probability
    predicted_class = tf.argmax(predictions[0]).numpy()
    predicted_class_name = Classes.get(predicted_class, 'Unknown')
    return predicted_class_name

@app.get("/")
def read_root():
    return FileResponse('index.html')


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    
    image = Image.open(io.BytesIO(await file.read()))
    image = image.resize((256, 256))
    prediction = getPrediction(image)
    #Add Timestamp to filename
    filename = f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}"
    image.save(f"images/{filename}")
    
    return {"filename": filename, "prediction": prediction}

if __name__ == "__main__":
    import uvicorn
    #run with reload
    uvicorn.run(app)