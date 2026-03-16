import cv2
import face_recognition
import numpy as np

# --- 第一步：載入已知的人臉資料 ---
# 這裡模擬從資料庫讀取員工照片
known_image = face_recognition.load_image_file("my_face.jpg") # 換成你的照片檔名
known_face_encoding = face_recognition.face_encodings(known_image)[0]

# 建立已知人臉特徵列表與對應名稱
known_face_encodings = [known_face_encoding]
known_face_names = ["User_A"] # 這裡填入員工姓名

# --- 第二步：啟動攝像頭 ---
video_capture = cv2.VideoCapture(0)

print("系統啟動中... 按下 'q' 鍵退出")

while True:
    # 擷取一幀畫面
    ret, frame = video_capture.read()
    if not ret:
        break

    # 為了加速，將影像縮小為 1/4 進行辨識
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    
    # OpenCV 預設是 BGR，要轉成 RGB 給 face_recognition 使用
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # 尋找畫面中所有的人臉位置與特徵
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # 比對臉部特徵是否匹配
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # 使用歐幾里得距離找出最接近的人臉
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # 將座標放大回原本的尺寸 (因為前面縮小了 4 倍)
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # 在人臉周圍畫框
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        # 在框下方顯示名字
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

    # 顯示結果視窗
    cv2.imshow('Face Recognition Clock-in System', frame)

    # 按下 'q' 鍵退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放資源
video_capture.release()
cv2.destroyAllWindows()
