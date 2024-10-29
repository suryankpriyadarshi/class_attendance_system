'''
Code for face embedding (getting numerical values related to a face in the form of a vector).
'''
import cv2 as cv
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN
import os
from keras_facenet import FaceNet
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

'''
Class for automating the face embedding.
Used for preprocessing as well as for real time face embedding.
'''
class FACELOADING:
  def __init__(self,directory): 
    '''
    Directory consisting of training data is passed for preprocessing. 
    '''
    self.directory=directory
    self.target_size=(160,160) # Target size set for FaceNet
    self.X=[] # To store face numericals
    self.Y=[] # To store corresponding labels
    self.detector=MTCNN()

  def extract_face(self,filename):
    '''
    Image file is passed for face extraction.
    '''
    img=cv.imread(filename)
    img=cv.cvtColor(img,cv.COLOR_BGR2RGB)
    result=self.detector.detect_faces(img) # MTCNN used for detecting faces
    x,y,w,h=result[0]['box']
    x,y=abs(x),abs(y)
    face=img[y:y+h,x:x+w]
    face_arr=cv.resize(face,self.target_size) # Resizing face for FaceNet
    return face_arr

  def load_faces(self,dir):
    '''
    Loads all faces from a specified directory containing multiple images of a person.
    '''
    FACES=[]
    for im_name in os.listdir(dir):
      try:
        path=dir+im_name
        single_face=self.extract_face(path) # Face extracted from an image
        FACES.append(single_face) # Each extracted face image is stored 
      except Exception as e:
        pass
    return FACES

  def load_classes(self):
    '''
    Loads all faces and their labels from subdirectories within the main directory, 
    where each subdirectory represents a person.
    '''
    for sub_dir in os.listdir(self.directory): # Loops through each sub-directory
      path=self.directory+'/'+sub_dir+'/'
      FACES=self.load_faces(path) # Loads faces for the current sub-directory
      labels=[sub_dir for _ in range(len(FACES))] # For each face loaded the label is added as the sub-directory name
      print(f'LOADED SUCCESSFULLY : {len(labels)}')
      self.X.extend(FACES) # Appends face data to X
      self.Y.extend(labels) # Append labels to Y
    return np.asarray(self.X), np.asarray(self.Y)

  def plot_images(self):
    '''
    Function to plot all face images stored in self.X for visual verification.
    '''
    for num,img in enumerate(self.X):
      ncols=3
      nrows=len(self.Y)//ncols+1
      plt.subplot(nrows,ncols,num+1)
      plt.imshow(img)
      plt.axis('off')


def get_embedding(face_img):
  '''
  Generates an embedding vector (numerical representation) for a given face image using FaceNet.
  '''
  embedder = FaceNet()
  face_img=face_img.astype('float32')
  face_img=np.expand_dims(face_img,axis=0)
  yhat=embedder.embeddings(face_img)
  return yhat[0]



if __name__ == '__main__':
    '''
    To pass section name (main directory name) as a command line argument.
    '''
    parser = argparse.ArgumentParser(description='Load faces and generate embeddings for a class attendance system.')
    parser.add_argument('section', type=str, help='Section name for the dataset (positional argument)')
    args = parser.parse_args()
    section = args.section

    current_dir = os.path.dirname(os.path.abspath(__file__)) # Path to current directory obtained
    dataset_dir = os.path.join(current_dir, 'dataset', 'Train', section) # Path to section directory created

    faceloading = FACELOADING(dataset_dir) # Faceloading object created for section directory
    X, Y = faceloading.load_classes() # Faces and corresponding labels are extracted
    EMBEDDED_X = []
    for img in X:
        EMBEDDED_X.append(get_embedding(img)) # For each loaded face FaceNet embeddings are generated
    
    '''
    EMBEDDED_X contains numerical embedding for each face data of a section.
    Y contains labels (names) corresponding to each face.
    '''
    EMBEDDED_X=np.array(EMBEDDED_X)
    EMBEDDED_X=EMBEDDED_X.tolist()
    Y = Y.tolist()

    '''
    Embedded data along with section name is stored to the database.
    '''

    mongodbUri = os.getenv('MONGODB_URI')
    client = MongoClient(mongodbUri)
    db = client['ClassAttendance']
    collection = db['sections']
    
    collection.update_one(
        {"section": section},
        {"$set": {"embeddings": EMBEDDED_X, "labels": Y}},
        upsert=True
    )