import cv2
import numpy as np
import time

#Set up YOLOv3 architecture
yolo_type = '320'
path = 'yolov3/yolov3-'+ yolo_type 
net = cv2.dnn.readNet(path + '.weights', path + '.cfg') #yolov3 320x320
#Try using GPU
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

#Get classes' names, colors and set the area treshold
with open('coco.names', 'r') as f:
    classes = f.read().strip().split('\n')

colors = np.random.uniform(0, 255, size=(len(classes), 3))
area_treshold = 0.2*np.ones((len(classes),))

#Resizes the image for the detection algorithm
def preprocess_image(image):
    blob = cv2.dnn.blobFromImage(image, 1.0 / 255.0, (320, 320), swapRB=True, crop=False)
    return blob

#Processes the image through the CNN
def forward_model(image):
    blob = preprocess_image(image)
    net.setInput(blob)
    # Perform forward pass and get output
    return net.forward(net.getUnconnectedOutLayersNames())

#Postprocess YOLO outputs to generate the bounding boxes
def postprocess_outputs(outputs, image_shape):
    conf_threshold = 0.55
    nms_threshold = 0.4
    image_width, image_height = image_shape

    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > conf_threshold:
                center_x, center_y, width, height = (detection[0:4] * np.array([image_height, image_width, image_height, image_width ])).astype('int')
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)
                boxes.append([x, y, width, height])
                confidences.append(confidence)
                class_ids.append(class_id)

    # Apply non-maximum suppression to remove overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    return indices, boxes, confidences, class_ids


def obstacle_and_bounding_boxes(postprocessed_tuple, image):
    indices, boxes, confidences, class_ids = postprocessed_tuple
    image_area = image.shape[0]*image.shape[1]
    clear = True #is the front clear?

    for i in indices:
        x, y, width, height = boxes[i]
        label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
        color = colors[class_ids[i]]
        cv2.rectangle(image, (x, y), (x + width, y + height), color, thickness=2)
        cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        area_covered_class = width*height/(image_area)
        print(f'Class {classes[class_ids[i]]} covers {(width, height)} {width*height} pixels, '
              f'i.e. {100*area_covered_class:.{3}}% of the image.')
        if area_covered_class > area_treshold[class_ids[i]]:
            clear = False
    return clear, image


#Unifies the execution of the functions for an image
def pipeline(image):
    outputs = forward_model(image)
    post_tuple = postprocess_outputs(outputs, image.shape[0:2])
    return obstacle_and_bounding_boxes(post_tuple, image)



if __name__=='__main__':
    image = cv2.imread('boat.jpg')
    print(f'is clear? {pipeline(image)[0]}')
    cv2.imshow('Object Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        t = time.time()
        clear, img = pipeline(frame)
        print(f'elapsed {time.time() - t}')
        cv2.imshow('Camera Feed', img)

        # Check for the 'q' key to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    
    
