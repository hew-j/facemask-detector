# import the libraries
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2

# Load the serialized face detector model from disk
prototxtPath = r"C:\Users\Hew J\PycharmProjects\pythonProject\face_detector\deploy.prototxt"
weightsPath = r"C:\Users\Hew J\PycharmProjects\pythonProject\face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# Load the face mask detector model from disk
print("[INFO] Loading face mask detector model...")
maskNet = load_model("mask_detector.model")

# Load the input image from disk, clone it, and grab the image spatial dimensions
path = r"C:\Users\Hew J\PycharmProjects\pythonProject\khj.jpg"
image = cv2.imread(path)
orig = image.copy()
(h, w) = image.shape[:2]

# Construct a blob from the image
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))

# Pass the blob through the network and obtain the face detections
print("[INFO] Computing face detections...")
faceNet.setInput(blob)
detections = faceNet.forward()

# Loop over the detections
for i in range(0, detections.shape[2]):

	# Extract the confidence (i.e., probability) associated with the detection
	confidence = detections[0, 0, i, 2]

	# Filter out weak detections by ensuring the confidence is greater than the minimum confidence
	if confidence > 0.5:
		# Compute the (x, y)-coordinates of the bounding box for the object
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")
		# Ensure the bounding boxes fall within the dimensions of the frame
		(startX, startY) = (max(0, startX), max(0, startY))
		(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

		# Extract the face ROI, convert it from BGR to RGB channel ordering, resize it to 224x224, and preprocess it
		face = image[startY:endY, startX:endX]
		face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
		face = cv2.resize(face, (224, 224))
		face = img_to_array(face)
		face = preprocess_input(face)
		face = np.expand_dims(face, axis=0)

		# Pass the face through the model to determine if the face has a mask or not
		(mask, withoutMask) = maskNet.predict(face)[0]

		# Determine the class label and color we'll use to draw the bounding box and text
		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

		# Include the probability in the label
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# Display the label and bounding box rectangle on the output frame
		cv2.putText(image, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

# Show the output image
cv2.imshow("Output", image)
cv2.waitKey(0)

# Cleanup
cv2.destroyAllWindows()