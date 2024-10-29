'''
Server code.
'''
import os
import numpy as np
import base64
import cv2 as cv
from flask import Flask, render_template, request, redirect, url_for, session
from record import match_faces,mark_attendence,attendence
from model import trainer
import time
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY')

'''
Connecting database.
'''
mongodb_uri = os.getenv('MONGODB_URI')
client = MongoClient(mongodb_uri)
db = client['ClassAttendance']

@app.route('/', methods=['GET']) # Base route used for login page
@app.route('/login', methods=['GET'])
def login():
    if 'username' in session: # If user is already looged in for the session redirect to home
        return redirect(url_for('home'))
    return render_template('login.html') # Otherwise redirect to login page

@app.route('/login/check', methods=['POST'])
def login_check():
    userid = request.form.get('Userid')  
    password = request.form.get('Password')

    collection = db['user_login']
    query = {"username": userid}
    document = collection.find_one(query)
    
    if document:
        if document['password'] == password: 
            session['username'] = userid # If password matches create new session for user 
            return redirect(url_for('home'))
        else:
            return render_template('login.html', error="Invalid password!") # Otherwise provide invalid password error
    else:
        return render_template('login.html', error="No such user exists") # If user does not exist provide invalid user error

@app.route('/home', methods=['GET'])
def home():
    if 'username' not in session: # If user not looged in redirect to login page
        return redirect(url_for('login'))
    username = session['username'] # If user logged in get the username from session data
    collection = db['user_login']
    query = {"username": username}
    document = collection.find_one(query)

    if document: # Get list of all sections for user 
        sections = document.get('sections', [])  
    else:
        sections = []  
    return render_template('home.html', sections=sections, userid = username) # Redirect to home page

@app.route("/attendance/<userid>/take_attendance",methods=['GET']) # Route to handle a request for taking attendance
def get_section(userid):
    section=request.args['section'] # Get the section from request
    return redirect(f'/attendance/{userid}/take_attendance/{section}') # Redirect to the next route

@app.route("/attendance/<userid>/take_attendance/<section>",methods=['GET']) # Route to display page for taking attendance
def take_attendence(userid,section):
    return render_template('attendance.html',userid=userid,section=section) 

@app.route("/attendance/<userid>/take_attendance/<section>/start_attendance",methods=['POST']) # Route for starting attendance
def start_attendence(userid,section):
    start=time.time()
    collection = db['sections']
    query = {"section":section}
    document=collection.find_one(query) # Get preprocessed details for the section
    if document and 'embeddings' in document:
        embeddings = document['embeddings']
    if document and 'labels' in document:
        labels = document['labels']
    
    embeddings=np.array(embeddings)
    labels=np.array(labels)

    my_model,encoder,attendence_matrix=trainer(embeddings,labels) # Train model over the preprocessed data
    
    image_data = request.form['image_data'] # Recieve base64 string image data
    # Decode the base64 string to recreate image
    img_data = base64.b64decode(image_data.split(',')[1])
    np_img = np.frombuffer(img_data, np.uint8)
    img = cv.imdecode(np_img, cv.IMREAD_COLOR)

    present=match_faces(img,my_model,encoder) # Send image and model for matching faces
    attendence_matrix=mark_attendence(attendence_matrix,present) # Mark attendance of each student
    attendence_list=attendence(section,attendence_matrix) # Get the attendance list
    end=time.time()
    print("time ",end-start)

    # Store attendance with current date, section name and techer's userid 
    current_date = datetime.now().date().isoformat()
    attendence_collection = db['attendance']
    attendance_filter = {
        'username':userid,
        'section':section,
        'date':current_date
    }
    attendance_query = {
        '$set':{
            'attendance':attendence_list
        }
    }
    result = attendence_collection.find_one_and_update(
    attendance_filter,
    attendance_query,
    upsert=True,               
    return_document=True)

    if result:
        print("Document found and updated or inserted.")
    else:
        print("Document not found, inserted new one.")

    return render_template('table.html',student_list=attendence_list)

@app.route("/attendance/<userid>/view_history",methods=['GET']) # Route for viewing history
def view_history(userid):
    section = request.args.get('section')
    collection = db['attendance']
    result =list( collection.find({
        'username':userid,
        'section':section
    }).sort('date', -1))
    return render_template('view.html',history=result)

@app.route('/attendance/<userid>/view_details', methods=['GET']) # Route for viewing attendance sheet of a particular section for a particular day
def view_attendance_details(userid):
    section = request.args.get('section')
    date = request.args.get('date')
    collection = db['attendance']
    query = {
        'username':userid,
        'section':section,
        'date':date
    }
    result = collection.find_one(query)
    attendence_list = list(result['attendance'])
    return render_template('table.html',student_list=attendence_list)


@app.route("/attendance/<userid>/edit",methods=['GET']) # Route for viewing history to edit
def edit(userid):
    section = request.args.get('section')
    collection = db['attendance']
    result =list( collection.find({
        'username':userid,
        'section':section
    }).sort('date', -1))
    return render_template('edit.html',history=result)
    
@app.route('/attendance/<userid>/edit_details', methods=['GET']) # Route for editing attendance sheet of a particular section for a particular day
def edit_attendance_details(userid):
    section = request.args.get('section')
    date = request.args.get('date')
    collection = db['attendance']
    query = {
        'username':userid,
        'section':section,
        'date':date
    }
    result = collection.find_one(query)
    attendence_list = list(result['attendance'])
    return render_template('edit_table.html',student_list=attendence_list,userid=userid,section=section,date=date)

@app.route('/attendance/<userid>/save_details', methods=['POST']) # Route for saving edited details
def save_attendance_details(userid):
    section = request.form.get('section')
    date = request.form.get('date')
    collection = db['attendance']
    query = {
        'username': userid,
        'section': section,
        'date': date
    }
    
    result = collection.find_one(query)
    attendence_list = list(result['attendance'])
    for student in attendence_list:
        roll_no = student['rollno']
        new_attendance = request.form.get(f'attendance_{roll_no}')
        if new_attendance:
            student['attendence'] = new_attendance
    update_query = {
        '$set': {'attendance': attendence_list}
    }
    collection.update_one(query, update_query)
    return redirect(url_for('home'))



@app.route('/logout', methods=['GET']) # Logout
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

if __name__ == "__main__":
    app.run(debug=True, port=3000)
