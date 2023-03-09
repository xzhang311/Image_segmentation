import cv2
import numpy as np
from PIL import Image


def instance_seg(img_file):
    path_to_model = 'frozen_inference_graph_coco.pb'
    path_to_config = 'mask_rcnn_inception_v2_coco_2018_01_28.pbtxt'

    model = cv2.dnn.readNetFromTensorflow(path_to_model, path_to_config)

    colors = np.random.randint(125, 255, (80, 3))

    # Choices
    segment_class_id = []

    if search == 'Person':
        segment_class_id = [0]
    elif search == 'Motorcycle':
        segment_class_id = [1]
    elif search == 'Car':
        segment_class_id = [2]
    elif search == 'All':
        segment_class_id = []

    if img_file is None:
        st.write("Error: could not read image")
        exit()
    else:
        img = np.asarray(img_file)
        img = cv2.resize(img, (650, 550), interpolation=cv2.INTER_LINEAR)
        height, width, _ = img.shape

        black_image = np.zeros((height, width, 3), np.uint8)
        black_image[:] = (0, 0, 0)
        blob = cv2.dnn.blobFromImage(img, swapRB=True)
        model.setInput(blob)
        boxes, masks = model.forward(["detection_out_final", "detection_masks"])
        detection_count = boxes.shape[2]

        count = 0
        for i in range(detection_count):
            box = boxes[0, 0, i]
            class_id = box[1]
            score = box[2]

            if score < thresh or (segment_class_id and class_id not in segment_class_id):
                continue
            x = int(box[3] * width)
            y = int(box[4] * height)
            x2 = int(box[5] * width)
            y2 = int(box[6] * height)

            roi = black_image[y: y2, x: x2]
            roi_height, roi_width, _ = roi.shape
            mask = masks[i, int(class_id)]
            mask = cv2.resize(mask, (roi_width, roi_height))
            _, mask = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)
            cv2.rectangle(img, (x, y), (x2, y2), (255, 0, 0), 3)
            contours, _ = cv2.findContours(np.array(mask, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            color = colors[int(class_id)]
            for cnt in contours:
                cv2.fillPoly(roi, [cnt], (int(color[0]), int(color[1]), int(color[2])))
            count += 1
            # cv2.imshow("Final", np.hstack([img, black_image]))
            # cv2.imshow("Overlay_image", ((0.6 * black_image) + (0.4 * img)).astype("uint8"))
        if count == 0:
            st.write("No objects found.")
        else:
            st.image(np.hstack([img, black_image]), channels="BGR", caption="Instance Segmentation")
            st.text(f"No of segmented objects : {count}")
            cv2.waitKey(0)


def semantic_seg(img):
    global segment_class_id
    import cv2
    import numpy as np

    # Load the model and configuration files
    path_to_model = 'frozen_inference_graph_coco.pb'
    path_to_config = 'mask_rcnn_inception_v2_coco_2018_01_28.pbtxt'

    model = cv2.dnn.readNetFromTensorflow(path_to_model, path_to_config)

    segment_class_id = []
    # Choose the class to segment
    if search == 'Person':
        segment_class_id = [0]
    elif search == 'Motorcycle':
        segment_class_id = [1]
    elif search == 'Car':
        segment_class_id = [2]

    # Load the input image
    img_file = img
    img_file = np.asarray(img_file)
    # Resize the image for faster processing
    img_file = cv2.resize(img_file, (650, 550), interpolation=cv2.INTER_LINEAR)
    height, width, _ = img_file.shape

    # Create a black image with the same size as the input image
    black_image = np.zeros((height, width, 3), np.uint8)
    black_image[:] = (0, 0, 0)

    # Preprocess the input image
    blob = cv2.dnn.blobFromImage(img_file, swapRB=True)
    model.setInput(blob)

    # Forward pass through the model
    boxes, masks = model.forward(["detection_out_final", "detection_masks"])
    detection_count = boxes.shape[2]

    # Iterate over the detected objects
    for i in range(detection_count):
        box = boxes[0, 0, i]
        class_id = box[1]
        score = box[2]

        # Filter out the objects that do not match the chosen class or have a low score
        if score < thresh or class_id != segment_class_id:
            continue

        # Get the bounding box coordinates and extract the ROI from the black image
        x = int(box[3] * width)
        y = int(box[4] * height)
        x2 = int(box[5] * width)
        y2 = int(box[6] * height)
        roi = black_image[y:y2, x:x2]

        # Resize the mask to match the size of the ROI
        mask = masks[i, int(class_id)]
        mask = cv2.resize(mask, (roi.shape[1], roi.shape[0]))

        # Threshold the mask and apply the bitwise AND operation to extract the segmented object
        _, mask = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)
        mask = mask.astype(np.uint8)
        img = cv2.bitwise_and(img_file[y:y2, x:x2], img_file[y:y2, x:x2], mask=mask)

        # Fill the segmented object with a random color
        color = np.random.randint(0, 255, (3,))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            cv2.fillPoly(roi, [cnt], (int(color[0]), int(color[1]), int(color[2])))

    # Display the segmented image
    st.image(np.hstack([img_file, black_image]), channels="BGR", caption="Semantic Segmentation")
    cv2.waitKey(0)


import time
import numpy as np
import streamlit as st

# Main
st.set_page_config(layout="wide")
with st.container():
    st.title("Image Segmentation")
    st.write("In digital image processing and computer vision, image segmentation is the process of partitioning a "
             "digital image into multiple image segments, also known as image regions or image objects (sets of "
             "pixels). The goal of segmentation is to simplify and/or change the representation of an image into "
             "something that is more meaningful and easier to analyze.[1][2] Image segmentation is typically used to "
             "locate objects and boundaries (lines, curves, etc.) in images. More precisely, image segmentation is "
             "the process of assigning a label to every pixel in an image such that pixels with the same label share "
             "certain characteristics.")

st.text("")
with st.container():
    st.header("Techniques of segmentation")
    st.subheader("Semantic segmentation")
    st.write("Semantic Segmentation follows three steps::")
    st.markdown("- Classifying: Classifying a certain object in the image.")
    st.markdown("- Localizing: Finding the object and drawing a bounding box around it.")
    st.markdown(
        "- Segmentation: Grouping the pixels in a localized image by creating a segmentation mask.Essentially, "
        "the task of Semantic Segmentation can be referred to as classifying a certain class of image and separating it from the rest "
        "of the image classes by overlaying it with a segmentation mask.")
    st.write("\nIt can also be thought of as the classification of images at a pixel level.")

    st.markdown('''
    <style>
    [data-testid="stMarkdownContainer"] ul{
        padding-left:40px;
    }
    </style>
    ''', unsafe_allow_html=True)

    st.subheader("Instance segmentation")
    st.write("Instance segmentation employs techniques from both semantic segmentation as well as object detection. "
             "Given an image we want to predict the location and identity of objects in that image (similar to object "
             "detection), however, rather than predicting bounding box for those objects we want to predict whole "
             "segmentation mask for those objects i.e which pixel in the input image corresponds to which object "
             "instance road.")

with st.container():
    st.text(" ")
    st.subheader("Implementation")
    st.write("From the sidebar import image and select any method,algorithm to perform segmentation on your image")
    # from PIL import Image
    #
    # img = Image.open("6.png")
    # st.image(img, caption="Original Image", width=650)

# Sidebar

# Model import
with st.container():
    st.sidebar.header("Import image: ")
    file = st.sidebar.file_uploader("Upload your file:", ['png', 'jpg', 'jpeg'], accept_multiple_files=False)

    if st.sidebar.button("Import"):
        with st.spinner("Importing image...."):
            time.sleep(5)
        st.sidebar.success("Imported successfully")
        st.image(file)

# Search for

with st.container():
    st.sidebar.header("Search for: ")
    search = st.sidebar.selectbox("Object", ("Person", "Car", "Motorcycle", "All"))

# Segmentation technique
with st.container():
    st.sidebar.header("Method")
    method = st.sidebar.selectbox("Please select any one method", ("Semantic", "Instance"))

# Model

with st.container():
    st.sidebar.header("Threshold: ")
    thresh = st.sidebar.select_slider("Confidence Threshold", options=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    algo = st.sidebar.selectbox("Algorithm", options=["Mask_RCNN"])
    if st.sidebar.button("OK"):
        if file is None:
            st.sidebar.error("Please upload an image first.")

        else:
            with st.spinner("Performing Segmentation...."):
                time.sleep(10)
            st.sidebar.success("Done ! ")

            if method == "Semantic" and algo == "Mask_RCNN":

                image = Image.open(file)

                semantic_seg(image)

            elif method == "Instance" and algo == "Mask_RCNN":

                image = Image.open(file)
                instance_seg(image)

            else:
                st.text("")

# Code
if st.checkbox("Show / Hide Code"):
    with st.container():
        st.code('''

def instance_seg(img_file):
    path_to_model = 'frozen_inference_graph_coco.pb'
    path_to_config = 'mask_rcnn_inception_v2_coco_2018_01_28.pbtxt'

    model = cv2.dnn.readNetFromTensorflow(path_to_model, path_to_config)

    colors = np.random.randint(125, 255, (80, 3))

    # Choices
    segment_class_id = []

    if search == 'Person':
        segment_class_id = [0]
    elif search == 'Motorcycle':
        segment_class_id = [1]
    elif search == 'Car':
        segment_class_id = [2]
    elif search == 'All':
        segment_class_id = []

    if img_file is None:
        st.write("Error: could not read image")
        exit()
    else:
        img = np.asarray(img_file)
        img = cv2.resize(img, (650, 550), interpolation=cv2.INTER_LINEAR)
        height, width, _ = img.shape

        black_image = np.zeros((height, width, 3), np.uint8)
        black_image[:] = (0, 0, 0)
        blob = cv2.dnn.blobFromImage(img, swapRB=True)
        model.setInput(blob)
        boxes, masks = model.forward(["detection_out_final", "detection_masks"])
        detection_count = boxes.shape[2]

        count = 0
        for i in range(detection_count):
            box = boxes[0, 0, i]
            class_id = box[1]
            score = box[2]

            if score < thresh or (segment_class_id and class_id not in segment_class_id):
                continue
            x = int(box[3] * width)
            y = int(box[4] * height)
            x2 = int(box[5] * width)
            y2 = int(box[6] * height)

            roi = black_image[y: y2, x: x2]
            roi_height, roi_width, _ = roi.shape
            mask = masks[i, int(class_id)]
            mask = cv2.resize(mask, (roi_width, roi_height))
            _, mask = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)
            cv2.rectangle(img, (x, y), (x2, y2), (255, 0, 0), 3)
            contours, _ = cv2.findContours(np.array(mask, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            color = colors[int(class_id)]
            for cnt in contours:
                cv2.fillPoly(roi, [cnt], (int(color[0]), int(color[1]), int(color[2])))
            count += 1
            # cv2.imshow("Final", np.hstack([img, black_image]))
            # cv2.imshow("Overlay_image", ((0.6 * black_image) + (0.4 * img)).astype("uint8"))

        st.image(np.hstack([img, black_image]), channels="BGR", caption="Instance Segmentation")
        st.text(f"No of segmented objects : {count}")
        cv2.waitKey(0)


def semantic_seg(img):
    global segment_class_id
    import cv2
    import numpy as np

    # Load the model and configuration files
    path_to_model = 'frozen_inference_graph_coco.pb'
    path_to_config = 'mask_rcnn_inception_v2_coco_2018_01_28.pbtxt'

    model = cv2.dnn.readNetFromTensorflow(path_to_model, path_to_config)

    segment_class_id = []
    # Choose the class to segment
    if search == 'Person':
        segment_class_id = [0]
    elif search == 'Motorcycle':
        segment_class_id = [1]
    elif search == 'Car':
        segment_class_id = [2]

    # Load the input image
    img_file = img
    img_file = np.asarray(img_file)
    # Resize the image for faster processing
    img_file = cv2.resize(img_file, (650, 550), interpolation=cv2.INTER_LINEAR)
    height, width, _ = img_file.shape

    # Create a black image with the same size as the input image
    black_image = np.zeros((height, width, 3), np.uint8)
    black_image[:] = (0, 0, 0)

    # Preprocess the input image
    blob = cv2.dnn.blobFromImage(img_file, swapRB=True)
    model.setInput(blob)

    # Forward pass through the model
    boxes, masks = model.forward(["detection_out_final", "detection_masks"])
    detection_count = boxes.shape[2]

    # Iterate over the detected objects
    for i in range(detection_count):
        box = boxes[0, 0, i]
        class_id = box[1]
        score = box[2]

        # Filter out the objects that do not match the chosen class or have a low score
        if score < thresh or class_id != segment_class_id:
            continue

        # Get the bounding box coordinates and extract the ROI from the black image
        x = int(box[3] * width)
        y = int(box[4] * height)
        x2 = int(box[5] * width)
        y2 = int(box[6] * height)
        roi = black_image[y:y2, x:x2]

        # Resize the mask to match the size of the ROI
        mask = masks[i, int(class_id)]
        mask = cv2.resize(mask, (roi.shape[1], roi.shape[0]))

        # Threshold the mask and apply the bitwise AND operation to extract the segmented object
        _, mask = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)
        mask = mask.astype(np.uint8)
        img = cv2.bitwise_and(img_file[y:y2, x:x2], img_file[y:y2, x:x2], mask=mask)

        # Fill the segmented object with a random color
        color = np.random.randint(0, 255, (3,))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            cv2.fillPoly(roi, [cnt], (int(color[0]), int(color[1]), int(color[2])))

    # Display the segmented image
    st.image(np.hstack([img_file, black_image]), channels="BGR", caption="Semantic Segmentation")
    cv2.waitKey(0)
        ''')
