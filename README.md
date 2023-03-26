# Object detection and segmentation

---

## App structure
![](pictures/app_diagram.png)


## CLI:
Train model:

    python cli.py train --dataset=<dataset path>

Evaluate model:

    python cli.py evaluate --dataset=<dataset path>

Run real-time demo with provided image or video file:

    python cli.py demo --file=<file path>


## API

Basic information (creator info, metrics, docker build datetime):

    GET /api/info

---

Open page for uploading your video or picture:

    GET /api/index

After upload, you'll be redirected to **/api/demo/<file name>**

---

Page with real-time demo on your file:
    
    GET /api/demo/<file name>


## Technologies

- Python
- OpenCV
- Flask


## Creator
*Eduard Khusnutdinov, Tomsk State University, 3d grade student*


