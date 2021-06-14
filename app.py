import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time
import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from flask import Flask, render_template, Response


app = Flask(__name__)


mixer.init()
sound = mixer.Sound('alarm.wav')


face = cv2.CascadeClassifier("haar cascade files\haarcascade_frontalface_alt.xml")
leye = cv2.CascadeClassifier("haar cascade files\haarcascade_lefteye_2splits.xml")
reye = cv2.CascadeClassifier("haar cascade files\haarcascade_righteye_2splits.xml")

lbl = ['Close', 'Open']

model = load_model('abcnnCat2.h5')
path = os.getcwd()
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
now = datetime.datetime.now()

def gen_frames():
    count = 0
    score = 0
    thicc = 2
    rpred = [99]
    lpred = [99]

    while True:

        ret, frame = cap.read()
        height, width = frame.shape[:2]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
        left_eye = leye.detectMultiScale(gray)
        right_eye = reye.detectMultiScale(gray)

        cv2.rectangle(frame, (0, height-50), (200, height), (0, 0, 0), thickness=cv2.FILLED)
        #cv2.putText(frame,'Current time:'+now, (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        #cv2.putText(frame, "Current time: "+(now.strftime("%Y-%m-%d %H:%M:%S")), (60, 20), font, 1,
         #           (255, 255, 255), 1, cv2.LINE_AA)

        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,255,100) , 1 )
        cv2.rectangle(frame, (3, 3), (width-3, 35), (0, 0, 0), thickness=cv2.FILLED)
        datet = str(datetime.datetime.now())
        frame = cv2.putText(frame,"Date & time: "+datet, (60, 25), font, 1, (250, 250, 200), 1, cv2.LINE_AA)

        for (x,y,w,h) in right_eye:
            r_eye=frame[y:y+h,x:x+w]
            count= count+1
            r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
            r_eye = cv2.resize(r_eye,(24,24))
            r_eye= r_eye/255
            r_eye=  r_eye.reshape(24,24,-1)
            r_eye = np.expand_dims(r_eye,axis=0)
            rpred = model.predict_classes(r_eye)

            if rpred[0]==1:
                lbl = 'Open'
            if rpred[0]==0:
                lbl = 'Closed'
                break


        for (x,y,w,h) in left_eye:
            l_eye=frame[y:y+h,x:x+w]
            count = count+1
            l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)
            l_eye = cv2.resize(l_eye,(24,24))
            l_eye= l_eye/255
            l_eye=l_eye.reshape(24,24,-1)
            l_eye = np.expand_dims(l_eye,axis=0)
            lpred = model.predict_classes(l_eye)
            if(lpred[0]==1):
                lbl='Open'
            if(lpred[0]==0):
                lbl='Closed'
            break


        if(rpred[0]==0 and lpred[0]==0):
            score=score+1
            cv2.putText(frame,"Closed",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)


        # if(rpred[0]==1 or lpred[0]==1):
        else:
            score=score-1
            cv2.putText(frame,"Open",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)


        if(score<0):
            score=0
        cv2.putText(frame,'Score:'+str(score),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
        if(score>15):
            #person is feeling sleepy so we beep the alarm
            #cv2.imwrite(os.path.join(path,'image.jpg'),frame)
            #cv2.imwrite(os.path.join(path, 'static/screen/image.jpg'), frame)


            try:
                sound.play()

            except:  # isplaying = False
                pass
            if(thicc<16):
                thicc= thicc+2


            else:
                thicc=thicc-2
                if(thicc<2):
                    thicc=2
            cv2.rectangle(frame,(0,0),(width,height),(0,0,255),thicc)
            #cv2.imshow('frame',frame)
            #if cv2.waitKey(1) & 0xFF == ord('q'):
             #   break

            if(score>20):
                cv2.imwrite(os.path.join(path, 'static/screen/image.jpg'), frame)
                email_user = 'mail' # enter your mail-id instead of mail
                email_password = 'pswd' # enter password here instead of pswd
                email_send = 'destination' #enter destination mail-id instead of destination

                subject = 'Driver Sleeping'

                msg = MIMEMultipart()
                msg['From'] = email_user
                msg['To'] = email_send
                msg['Subject'] = subject

                body = 'Your Driver is under fatigue. Please see the attached image'
                msg.attach(MIMEText(body, 'plain'))

                filename = 'static/screen/image.jpg'
                attachment = open(filename, 'rb')

                part = MIMEBase('application', 'octet-stream')
                part.set_payload((attachment).read())
                encoders.encode_base64(part)
                part.add_header('Content-Disposition', "attachment; filename= " + filename)

                msg.attach(part)
                text = msg.as_string()
                server = smtplib.SMTP('smtp.gmail.com', 587)
                server.starttls()
                server.login(email_user, email_password)

                server.sendmail(email_user, email_send, text)
                server.quit()

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/second')
def second():
    return render_template('second.html')



@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary = frame')


if __name__ == '__main__':
    app.run(debug=True)
