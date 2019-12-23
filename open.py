import cv2
def generate():
	path1='D:/opencvsubject/haarcascade_frontalface_default.xml'
	path2='D:/opencvsubject/haarcascade_eye.xml'
	face_cascade=cv2.CascadeClassifier(path1)
	eye_cascade=cv2.CascadeClassifier(path2)
	camera=cv2.VideoCapture(0)
	count=0
	while(True):
		ret,frame=camera.read()
		gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		faces=face_cascade.detectMultiScale(gray,1.3,5)
		for(x,y,w,h) in faces:
			img=cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
			f=cv2.resize(gray[y:y+h,x:x+w],(200,200))
			cv2.imwrite('D:/opencvsubject/data/at/jm/%s.pgm' % str(count),f)
			count+=1
		cv2.imshow("camera",frame)
		if count==60:
		    break
		if cv2.waitKey(1000//12)& 0xff == ord("q"):
			break
	camera.release()
	cv2.destroyAllWindows()
if __name__=="__main__":
	generate()