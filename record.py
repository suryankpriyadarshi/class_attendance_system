'''
Utility functions for detecting faces, matching them to existing embeddings,
and marking attendance status for the attendance system.
'''
import cv2 as cv
import pandas as pd
from mtcnn.mtcnn import MTCNN
import os
import random
from embedder import get_embedding

def scan_image(img):
    '''
    Utilized to detect faces from a given image.
    '''
    detector = MTCNN()
    mid_height, mid_width = img.shape[0] // 2, img.shape[1] // 2
    face_boxes = []
    for _ in range(5):  # Image scanned multiple times for ensuring all faces are detected
        mid_y, mid_x = random.randint(mid_height - 300, mid_height + 300), random.randint(mid_width - 300, mid_width + 300)
        
        # Each time image is divided randomly into four parts for ensuring better detection
        img1 = img[0:mid_y, 0:mid_x] 
        img2 = img[0:mid_y, mid_x:len(img[0])]
        img3 = img[mid_y:len(img), 0:mid_x]
        img4 = img[mid_y:len(img), mid_x:len(img[0])]
        
        # Detect faces only if the image section has non-zero dimensions
        if img1.shape[0] > 0 and img1.shape[1] > 0:
            output1 = detector.detect_faces(img1)
            face_boxes.extend([item['box'] for item in output1])
        
        if img2.shape[0] > 0 and img2.shape[1] > 0:
            output2 = detector.detect_faces(img2)
            for item in output2:
                item['box'][0] += mid_x
                face_boxes.append(item['box'])
        
        if img3.shape[0] > 0 and img3.shape[1] > 0:
            output3 = detector.detect_faces(img3)
            for item in output3:
                item['box'][1] += mid_y
                face_boxes.append(item['box'])
        
        if img4.shape[0] > 0 and img4.shape[1] > 0:
            output4 = detector.detect_faces(img4)
            for item in output4:
                item['box'][0] += mid_x
                item['box'][1] += mid_y
                face_boxes.append(item['box'])
                
    return face_boxes  # Bounding box data for each face returned


def match_faces(image,my_model,encoder):
    img=cv.cvtColor(image,cv.COLOR_BGR2RGB)
    face_boxes=scan_image(img) # Get bounding boxes
    present=[]
    for box in face_boxes:
        x,y,w,h=box
        face=img[y:y+h,x:x+w]
        face=cv.resize(face,(160,160))
        test_face=get_embedding(face) # Get face embedding
        test_face=[test_face]
        ypreds=my_model.predict(test_face) # Predict test face
        target=encoder.inverse_transform(ypreds)
        present.append(target[0])
    present.sort()
    return present


def mark_attendence(attendence_matrix,present):
    for student in attendence_matrix:
        if present.count(student[0]) > 3: # If student was found present 3 times out of the  five scans mark present
                student[1]='P'
    return attendence_matrix

def attendence(section,attendence_matrix):
    attendence_sheet=pd.DataFrame(attendence_matrix,columns =['NAME', 'ATTENDENCE'])
    attendence_array=attendence_sheet.values.tolist()
    attendence_list=[]
    # Making the proper attendance list with serial number and roll number
    for i,student in enumerate(attendence_array):
        temp={
            "sl_no":i+1,
            "name":student[0],
            "rollno":f'{i + 1:03}',
            "attendence":student[1]
        }
        attendence_list.append(temp)
    return attendence_list

    
    
