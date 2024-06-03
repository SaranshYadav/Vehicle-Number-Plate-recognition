# Real-Time-Number-Plate-Recognition


## Tech Stack
* [yolov4](https://github.com/theAIGuysCode/yolov4-custom-functions) : I used this OD model because it performs much better than traditional computer vision methods.
* [Easy OCR](https://github.com/JaidedAI/EasyOCR) : In this project I used EasyOCR to extract text and leverage a size filtering algorithm to grab the largest detection region. EasyOCR is build on PyTorch.
* [openCV](https://opencv.org/): It is a library mainly used at real-time computer vision.
* [Tensorflow](https://github.com/tensorflow/models) : Here I used Tensorflow object detection Model (SSD MobileNet V2 FPNLite 320x320) to detect the plate trained on a Kaggle Dataset.
* Python Libraries: Most of the libraries are mentioned in [requirements.txt] but some of the libraries and requirements depends on the user's machines, whether its installed or not and also the libraries for Tensorflow Object Detection (TFOD) consistently change. Eg: pycocotools, pytorch with CUDA acceleration (with or without GPU), microsoft visual c++ 19.0 etc.

## Steps
These outline the steps I used to go through in order to get up and running with ANPR. 

### Install and Setup :

<b>Step 1.</b> Create a new virtual environment 
<pre>
python -m venv arpysns
</pre> 
<br/>
<b>Step 2.</b> Activate your virtual environment
<pre>
source tfod/bin/activate # Linux
.\arpysns\Scripts\activate # Windows 
</pre>
<br/>
<b>Step 3.</b> Install dependencies and add virtual environment to the Python Kernel
<pre>
python -m pip install --upgrade pip
pip install ipykernel
python -m ipykernel install --user --name=anprsys
</pre>
<br/>

### Dataset: 
Used the [Car License Plate Detection](https://www.kaggle.com/andrewmvd/car-plate-detection) kaggel dataset and manually divided the collected images into two folders train and test so that all the images and annotations will be split among these two folders.

### Training Object Detection Model
I used pre-trained state-of-the-art model and just fine tuned it on our particular specific use case.Begin the training process by opening [Real Time Number Plate Detection] and installed the Tensoflow Object Detection (TFOD) 


* Visualization of Loss Metric, learning rate and number of steps:

<pre>
tensorboard --logdir=.
</pre>


### Detecting License Plates

![Screenshot 2021-12-10 130124](https://user-images.githubusercontent.com/56076028/145536393-986af131-ce84-4d4c-8174-735ed492a45b.jpg)


### Apply OCR to text

<pre>
import easyocr
detection_threshold=0.7
image = image_np_with_detections
scores = list(filter(lambda x: x> detection_threshold, detections['detection_scores']))
boxes = detections['detection_boxes'][:len(scores)]
classes = detections['detection_classes'][:len(scores)]
</pre>

![Screenshot 2021-12-10 125508](https://user-images.githubusercontent.com/56076028/145536427-d27c0fdc-cd30-446b-9b16-6408fdb4efcd.jpg)

### Results

Used this in real time to detect the license plate and stored the text in .csv file and images in the Detection_Images folder.

### Object Detection Metric:
![evaluation metric](https://user-images.githubusercontent.com/56076028/145684944-29306983-8396-47a2-9a08-f13a86d56f08.jpg)

![evaluation metric detail](https://user-images.githubusercontent.com/56076028/145684945-7f17e0b6-e623-4a71-b163-388a84d713fd.jpg)

<pre>
tensorboard --logdir=.
</pre>

