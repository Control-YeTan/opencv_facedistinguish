import cv2
import numpy as np
import os
import sys
filepath='D:/opencvsubject/data/at/'
def read_images(path='D:/opencvsubject/data/at/',sz=None):
	id=0
	X,y=[],[]
	for dirname,dirnames,filenames in os.walk(path):
		for subdirname in dirnames:
			subject_path=os.path.join(dirname,subdirname)
			for filename in os.listdir(subject_path):
				try:
					
					filename=os.path.join(subject_path,filename)
					im=cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
					X.append(np.asarray(im,dtype=np.uint8))
					y.append(id)				
				except:				
					raise
			id=id+1
	return [X,y]
def face_rec():
	names=['yt','zth','zzh','ckq']
	[X,y]=read_images(filepath)
	print(y)
	#model=cv2.face.EigenFaceRecognizer_create()效果不好
	model=cv2.face.LBPHFaceRecognizer_create()#效果好
	#model=cv2.face.FisherFaceRecognizer_create()效果不好
	model.train(np.asarray(X),np.asarray(y))
	camera=cv2.VideoCapture(0)
	face_cascade=cv2.CascadeClassifier('D:/opencvsubject/haarcascade_frontalface_default.xml')
	while(True):
		read,img=camera.read()
		faces=face_cascade.detectMultiScale(img,1.3,5)
		for(x,y,w,h) in faces:
			img=cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
			gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
			roi=gray[x:x+w,y:y+h]
			try:
				roi=cv2.resize(roi,(200,200),interpolation=cv2.INTER_LINEAR)
				params=model.predict(roi)
				print("label:%s,confidence:%.2f" % (params[0],params[1]))
				cv2.putText(img,names[params[0]],(x,y-20),cv2.FONT_HERSHEY_SIMPLEX,1,255,2)
				cv2.putText(img, str(np.around(params[1], decimals=2)), (x + 60, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
			except:
				continue
		cv2.imshow("camera",img)
		if cv2.waitKey(1000//12) & 0xff==ord("q"):
			break
	camera.release()		
	cv2.destroyAllWindows()
if __name__=="__main__":
		face_rec()