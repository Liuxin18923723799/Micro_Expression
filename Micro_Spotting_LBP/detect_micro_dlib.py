
from imutils import face_utils
import dlib
import cv2
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import os
import numpy as np 
import lbp
import math
import pandas as pd
from src import util
import visualization
import moviepy.editor as mp

shape_predictor = 'shape_predictor_68_face_landmarks.dat'
p68 = "model/shape_predictor_68_face_landmarks.dat"
p5 = "model/shape_predictor_5_face_landmarks.dat"
detector=dlib.get_frontal_face_detector()
predictor_68 = dlib.shape_predictor(p68)
predictor_5 = dlib.shape_predictor(p5)

OUT_FOLDER ='CASME2_cropped'
folder_video = 'CASME_A'
folder_section = 'sectionA'

CASME2_FILE = 'casme2.xls'


# previous function
def divide_image_to_block(gray_img):
	pos_x = [[27,52],[53,78],[79,104],[105,130]]
	pos_y = [[27,52],[53,78],[79,104],[105,130]]
	list_img_block = []
	wt = 0
	for xa in pos_x:
		for ya in pos_y:
			row_st = xa[0]
			row_en = xa[1]

			col_st = ya[0]
			col_en = ya[1]
			
			
			img_block = gray_img[col_st:col_en,row_st: row_en]

			list_img_block.append(img_block)
	return list_img_block

def shape_to_np(shape, dtype="int"):
	coords = np.zeros((5, 2), dtype=dtype)
	for i in range(0, 5):
	     coords[i] = (shape.part(i).x, shape.part(i).y)
	return coords

def calculate_distance_block( curr_img, hframe, tframe ):

	blocks_curr = divide_image_to_block(curr_img)
	blocks_head = divide_image_to_block(hframe)
	blocks_tail = divide_image_to_block(tframe)
	num_block = len(blocks_curr)
	dist_list = []
	
	for iblock in range(0,num_block):
		blck_curr = blocks_curr[iblock]
		blck_hfrm = blocks_head[iblock]
		blck_tfrm = blocks_tail[iblock]
		lbp_feat_1 = lbp.extract_LBP_feature(blck_curr)
	
		lbp_feat_2 = lbp.extract_LBP_feature(blck_hfrm)
		lbp_feat_3 = lbp.extract_LBP_feature(blck_tfrm)

		avg_feat = np.add(lbp_feat_2,lbp_feat_3)/2
		
		dist = ChiSquare_dist(lbp_feat_1,avg_feat)
			
		if (math.isnan(dist)):
			dist = 10.0
		dist_list.append(dist)

	dist_list.sort()
	
	sum_dist = 0.0
	for i in range(num_block-12,num_block):
		sum_dist = sum_dist + dist_list[i]
	return sum_dist
	

def detect_ME_by_LBP_dlib_video(video_file):
	print("Starting !!")
	cap =cv2.VideoCapture(video_file)
	num_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	Farr = np.zeros(num_frame)

	Carr = np.zeros(num_frame)
	processing_list = []
	count = 0
	idx = 0
	st_idx = 0
	en_idx = 0

	window_len = 17
	L = int(window_len / 2)
	detected_face = 0
	pre_shape = 0
	while (cap.isOpened()):
		ret, img = cap.read()
		if (ret ==False):
			break
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		idx = idx + 1
		dets = detector(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),2)
		
		if (len(dets) > 0):
			iface = 0
			face_distance = 0
			for (j,d) in enumerate(dets):
				if (d.right() - d.left() > face_distance):
					face_distance = d.right() - d.left()
					iface = j
			detected_face = dets[iface]
			#shape68 = predictor_68(img,detected_face )
			shape5 = predictor_5(img, detected_face)
			#shape68 = face_utils.shape_to_np(shape68)
			
			if (idx-1==0 or detected_face == 0):
				pre_shape = shape5
				detected_face = 1

			curr_shape = shape5
			curr_shape = util.process_dlib_shape(pre_shape, curr_shape)

			alignedFace = dlib.get_face_chip(img,curr_shape, 200 ,0.2)
			

			#for (x, y) in shape5:
				#cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
			#util.write_out_img(alignedFace ,frm_idx , 'out')
			pre_shape = curr_shape


			processing_list.append(alignedFace)
			
			count =  len(processing_list)
			if (count > window_len):
				processing_list.pop(0)

			count =  len(processing_list)
			L = int(window_len/2) 
			if (count == window_len):
				st_idx = idx - window_len + 1
				apex_idx = idx - int(window_len/2)

				hframe = processing_list[0]
				tframe = processing_list[count-1]

				curr_frame = processing_list[int(window_len/2)]

				dist = calculate_distance_block(curr_frame,hframe,tframe)

				pos =  idx - int(window_len/2) - 1
				Farr[pos] = dist 
	print("COMPLETE CALCULATING F ARRAY ")
	for i in range(L,num_frame-L):
		Carr[i] = Farr[i] - 0.5*(Farr[i-L] + Farr[i+L])
		if (Carr[i] < 0):
			Carr[i] = 0

	Cmean = np.mean(Carr)
	Cmax = np.max(Carr)
	epsilon = 0.55
	# threshold
	Thr = Cmean + epsilon * (Cmax -  Cmean)

	res = np.zeros(num_frame)
	apex_list = []
	print(Thr)
	print(Carr)
	for i in range(0,num_frame):
		if (Carr[i] >= Thr and i >= 30 and i <= num_frame - 30):
			res[i] = 1
			apex_list.append(i)
			j = i+1
			k = i-1
			count = 0
			while (count <= 5):
				count = count + 1
				res[j] = 1
				res[k] = 1
				j = j + 1
				k = k - 1

	
	cap.release()
	return res


def ChiSquare_dist( np1, np2):
	np_shape = np1.shape[0]
	dist = 0.0
	for i in range(0,np_shape):
		if (np1[i] + np2[i] == 0):
			xs = 0.0
		else:
			xs = (np1[i] - np2[i])*(np1[i] - np2[i]) / (np1[i] + np2[i])

		dist = dist + xs

	return dist


def main():

 
	# initialize dlib's face detector (HOG-based) and then create
	# the facial landmark predictor
	video_file = 'me_amstrong.mp4'

	#video_file = os.path.join(video_folder,video_file)

	#res_vector = detect_me_by_LBP(video_folder)
	res = detect_ME_by_LBP_dlib_video(video_file)

	print(res)
	visualization.visualize_micro_expression(video_file , res)

if __name__ == '__main__':
    main()  
