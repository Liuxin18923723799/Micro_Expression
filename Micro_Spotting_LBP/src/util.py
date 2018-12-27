
import os
import numpy as np
import cv2

from imutils import face_utils


def write_out_img(out_img,  frm_idx, folder = '', prefix = '' ):
	out_img_name = ''

	if (frm_idx <10):
		out_img_name = '00' + str(frm_idx) + '.jpg'
	else:
		if (frm_idx < 100):
			out_img_name = '0' + str(frm_idx) + '.jpg'
		else:
			out_img_name = str(frm_idx) + '.jpg'
	out_file_name = os.path.join(folder,out_img_name)


	cv2.imwrite(out_file_name,out_img)

def get_transform_matrix(src_pts, dst_pts):
	tfm = np.float32([[1,0,0],[0,1,0]])
	n_pts = src_pts.shape[0]
	ones = np.ones((n_pts,1),src_pts.dtype)
	src_pts_st = np.hstack([src_pts , ones])
	dst_ptx_st = np.hstack([dst_pts, ones])

	A, res,  rank ,singular = np.linalg.lstsq(src_pts_st ,dst_ptx_st )

	if (rank==3):
		tfm = np.float32( [[A[0,0], A[1,0], A[2, 0]], [A[0,1], A[1,1], A[2, 1]] ])
	elif (rank==2):
		tfm = np.float32( [  [A[0,0], A[1,0] , 0], [A[0,1], A[1,1] , 0] ])
	return tfm


def face_registration(img, five_points ):
	imgSize = (200,150)
	xx = five_points[:,0]
	yy = five_points[:,1]
	dst = np.array( (xx,yy) ).astype(np.float32)

	xt = np.array([ 105.0 ,125.0,  30.0 , 50.0 , 75.0])
	yt = np.array([85.0 , 85.0 ,85.0 ,85.0,  120.0 ])

	
	src = np.array( (xt,yt) ).astype(np.float32)
	src = np.transpose(src)
	dst = np.transpose(dst)



	#transmat = cv2.getAffineTransform(dst, src)
	tfm = get_transform_matrix(dst, src)
	gray_img = cv2.cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	alignedFace = cv2.warpAffine ( img, tfm, (160,160))
	return alignedFace

def detect_and_crop_face( img , pre_shape , i):
	dets = detector(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),2)
	if (len(dets) > 0):
		iface = 0
		face_distance = 0
		for (j,d) in enumerate(dets):
			if (d.right() - d.left() > face_distance):
				face_distance = d.right() - d.left()
				iface = j
		detected_face = dets[iface]
		shape68 = predictor_68(img,detected_face )
		shape5 = predictor_5(img, detected_face)
		shape68 = face_utils.shape_to_np(shape68)
		shape5 = face_utils.shape_to_np(shape5)
		if (i==0):
			pre_shape = shape5
		curr_shape = shape5
		curr_shape = util.process_moving_shape(pre_shape, curr_shape)

		alignedFace = util.face_registration(img, curr_shape)

		#for (x, y) in shape5:
			#cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
		#util.write_out_img(alignedFace ,frm_idx , 'out')
		pre_shape = curr_shape

	return alignedFace , pre_shape


def process_moving_shape(pre_shape , curr_shape):
	dst = np.sqrt(((pre_shape-curr_shape)**2).sum(-1)).sum(0)
	if (dst >= 5):
		return curr_shape
	else:
		return pre_shape

def process_dlib_shape(pre_shape , curr_shape):
	futil_pre_shape = face_utils.shape_to_np(pre_shape)
	futil_curr_shape = face_utils.shape_to_np(curr_shape)
	dst = np.sqrt(((futil_pre_shape-futil_curr_shape)**2).sum(-1)).sum(0)
	if (dst >= 5):
		return curr_shape
	else:
		return pre_shape

