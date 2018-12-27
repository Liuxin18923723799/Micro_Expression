import numpy as np 
import cv2
import dlib
import moviepy.editor as mp

shape_predictor = 'shape_predictor_68_face_landmarks.dat'
p68 = "model/shape_predictor_68_face_landmarks.dat"
p5 = "model/shape_predictor_5_face_landmarks.dat"
detector=dlib.get_frontal_face_detector()
predictor_68 = dlib.shape_predictor(p68)
predictor_5 = dlib.shape_predictor(p5)

def shape_to_np(shape, dtype="int"):
	coords = np.zeros((68, 2), dtype=dtype)
	for i in range(0, 68):
	     coords[i] = (shape.part(i).x, shape.part(i).y)
	return coords

def visualize_micro_expression(video_file , micro_positions):
	print(" start visualize ")
	cap =cv2.VideoCapture(video_file)
	num_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	face_size = 224
	window_len = 17
	count = 0
	Frames = []
	while (cap.isOpened()):
		ret, img = cap.read()
		if (ret ==False):
			break
		#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		#cv2.rectangle(img,(384,0),(510,128),(0,255,0),3)
		if (micro_positions[count] > 0):
			dets = detector(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),2)
			
			if (len(dets) > 0):
				iface = 0
				face_distance = 0
				for (j,d) in enumerate(dets):
					if (d.right() - d.left() > face_distance):
						face_distance = d.right() - d.left()
						iface = j
				detected_face = dets[iface]
				xy1 = ( detected_face.left() , detected_face.top())
				xy2 = ( detected_face.right() , detected_face.bottom())
				img2 = cv2.rectangle(img, xy1 , xy2 , (255,0,0), 1)
				print(xy1, xy2)
			else:
				img2 = img
		else:
			img2 = img
		frame = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
		Frames.append(frame)
		count = count + 1
	audioclip = mp.AudioFileClip(video_file)
	fpsCV = int(cap.get(cv2.CAP_PROP_FPS))

	print("FPS ",fpsCV)
	OUT_VIDEO_FILE = 'out_amstrong.mp4'
	video_clip = mp.ImageSequenceClip(Frames, fps=fpsCV)
	video_clip= video_clip.set_audio(audioclip)
	video_clip.write_videofile(OUT_VIDEO_FILE)







