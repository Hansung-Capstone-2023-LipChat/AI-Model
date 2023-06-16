# python recognize_faces_video.py --encodings encodings.pickle --display 1
# 필요한 패키지 가져오기
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

from imutils.video import VideoStream
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2

# Firebase database 인증 및 앱 초기화
cred = credentials.Certificate('./firebase_json/newlipchat-firebase-adminsdk-qjh2u-086df0f3dd.json')
firebase_admin.initialize_app(cred,
                              {"databaseURL": 'https://newlipchat-default-rtdb.firebaseio.com/'})
# 유저 uid를 받아오기
name = 'sang'

# 연산 직전
cal_ing_face = '1'
ref = db.reference('flag/' + name)
ref.update({'face': cal_ing_face})
faceid = "Unknown"

# 연산 중
# parsing을 구성하고 parsing 구문 입력
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings")
ap.add_argument("-y", "--display", type=int, default=1,
	help="whether or not to display output frame to screen")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
	help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

# 학습한 얼굴 및 얼굴 임베딩 로드
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())
# Video Stream 및 포인터 초기화. 비디오 파일을 출력 후 카메라 센서 동작
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(1.0)

ex_faceid = "0"
realfaceid = "Unknown"
decide = 0

# Video File Stream roof
while True:
	# 스레드된 Video Stream에서 프레임 가져오기
	frame = vs.read()

	# 입력 프레임을 BGR에서 RGB 변환 후 크기 조정 750px (처리 속도 향상)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	rgb = imutils.resize(frame, width=750)
	r = frame.shape[1] / float(rgb.shape[1])
	# 경계 상자 (X,Y) 좌표 감지
	# 입력 프레임의 각 얼굴에 해당하는 좌표 계산
	# 각 얼굴에 대한 임베딩 진행
	boxes = face_recognition.face_locations(rgb,
											model=args["detection_method"])
	encodings = face_recognition.face_encodings(rgb, boxes)
	faceids = []

	# 얼굴 임베딩 반복
	for encoding in encodings:
		# 입력 이미지의 각 얼굴을 알려진 것과 일치시키려고 시도
		# encodings
		matches = face_recognition.compare_faces(data["encodings"], encoding)
		#faceid = "Unknown"
		# 일치하는 항목을 찾았는지 확인
		if True in matches:
			# 일치하는 모든 얼굴의 인덱스를 찾은 다음
			# 각 얼굴이 나온 횟수를 계산한다.
			matchedIdxs = [i for (i, b) in enumerate(matches) if b]
			counts = {}
			# 반복해서 일치하는 인덱스를 찾고 인덱스가 나오는 횟수를 유지해서 얼굴을 각각
			# 인덱스에 대해서 인식합니다.
			for i in matchedIdxs:
				faceid = data["faceids"][i]
				counts[faceid] = counts.get(faceid, 0) + 1
			# 가장 많이 반복된 인덱스를 face id로 결정합니다.
			# 만약 전부 동점일 경우 인덱스의 첫번째 항목이
			# face id로 결정됩니다.
			faceid = max(counts, key=counts.get)

		# 이름 목록을 업데이트
		faceids.append(faceid)

		# 인식된 얼굴이 무엇인지 알려줌
		for ((top, right, bottom, left), faceid) in zip(boxes, faceids):
			# 좌표 지정
			top = int(top * r)
			right = int(right * r)
			bottom = int(bottom * r)
			left = int(left * r)
			# 예상된 얼굴을 영상에 사각형을 그려 표현함.
			cv2.rectangle(frame, (left, top), (right, bottom),
						  (0, 255, 0), 2)
			y = top - 15 if top - 15 > 15 else top + 15
			cv2.putText(frame, faceid, (left, y), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 255, 0), 2)

	# 출력 프레임을 화면에 표시하도록 함.
	if args["display"] > 0:
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF
		# faceid를 확정함.
		# 이후 face recognition을 마무리 지음
	if ex_faceid == faceid:
		decide += 1
		if decide == 10:
			# 연산 후
			realfaceid = faceid

			cal_ing_face = '0'
			ref = db.reference('flag/'+name)
			ref.update({'face': cal_ing_face,'current_face_id': realfaceid})
			break
	ex_faceid = faceid

# clear memory
cv2.destroyAllWindows()
vs.stop()


##
	영상을 5초 정도 녹화한다.
	영상의 파일 확장자는 .mp4이다. 
##