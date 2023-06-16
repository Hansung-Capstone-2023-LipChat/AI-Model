import cv2
import face_recognition
import pickle
import time

image_file = 'C:/Users/SangHyuk/face recog/example/1 (4).png'
encoding_file = 'encodings.pickle'
unknown_name = 'Unknown'
model_method = 'cnn-gpu'


def detectAndDisplay(image):
    start_time = time.time()
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 입력 이미지에서 각 얼굴에 해당하는 box 좌표를 감지하고 얼굴 임베딩 계산
    boxes = face_recognition.face_locations(rgb,
                                            model=model_method)
    encodings = face_recognition.face_encodings(rgb, boxes)

    # 감지된 각 얼굴의 이름 목록 초기화
    names = []

    # 얼굴 임베딩 반복
    for encoding in encodings:
        # 입력 이미지의 각 얼굴과 학습된 데이터 매치
        matches = face_recognition.compare_faces(data["encodings"],
                                                 encoding)
        name = unknown_name

        # 데이터가 매치된 경우
        if True in matches:
            # 일치하는 모든 얼굴의 인덱스를 찾고, 얼굴 일치 횟수 계산을 위한 초기화
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            # 일치하는 인덱스를 반복하고, 인식된 각 얼굴의 수 유지
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            # 가장 많은 표를 얻은 label 선택
            name = max(counts, key=counts.get)

        # 이름 목록 업데이트
        names.append(name)

    # 인식된 얼굴 반복
    for ((top, right, bottom, left), name) in zip(boxes, names):
        # 이미지에 인식된 얼굴의 box를 그림
        y = top - 15 if top - 15 > 15 else top + 15
        # 학습된 얼굴(인식한 얼굴)의 경우 녹색 선
        color = (0, 255, 0)
        line = 2
        # 학습되지 않은 얼굴(인식하지 못한 얼굴)의 경우 빨간색 선
        if (name == unknown_name):
            color = (0, 0, 255)
            line = 1
            name = ''
        # 사각형 그리기
        cv2.rectangle(image, (left, top), (right, bottom), color, line)
        y = top - 15 if top - 15 > 15 else top + 15
        # 텍스트 추가
        cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, color, line)

    end_time = time.time()
    # 소요시간 체크
    process_time = end_time - start_time
    print("=== A frame took {:.3f} seconds".format(process_time))
    cv2.imshow("Recognition", image)


# 학습된 데이터 load
data = pickle.loads(open(encoding_file, "rb").read())

# 테스트할 이미지 load
image = cv2.imread(image_file)
detectAndDisplay(image)

cv2.waitKey(0)
cv2.destroyAllWindows()