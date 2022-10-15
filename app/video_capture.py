from app import *

camera = cv2.VideoCapture(-1)
if not (camera.isOpened()):
    print("Could not open video device")
frame_width = int(camera .get(3))
frame_height = int(camera .get(4))

size = (frame_width, frame_height)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')  
#creating the instance for the video capturing to save



#function to get the frames to stream in webpage and store it in file
def generate_frames():
    while True:

        ## read the camera frame
        success, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
  
        # Detect the faces  
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)  
    
        # Draw the rectangle around each face  
        for (x, y, w, h) in faces:  
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)  


        if not success:
            break
        else:

            ret, buffer = cv2.imencode('.jpg', frame)
           
            frame = buffer.tobytes()



        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



# url routes to navigate pages
#render the first html page
@app.route('/image',methods=['POST','GET'])
def image():
    return render_template('video_stream.html')


#calling the video function to stream video in webpage
@app.route('/video',methods=['POST','GET'])
def video():

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

#function finish recording and saving it
@app.route('/finished',methods=['POST','GET'])
def finished():
    if request.method == 'POST':
        print(1)
        camera.release()

        cv2.destroyAllWindows()
    else:
        print(0)
    