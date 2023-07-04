import tensorflow as tf
import numpy as np
import json
import struct
from struct import unpack
import cv2
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tensorflow.keras.utils import to_categorical
import random


image_height = 256
image_width = 256
num_classes = 2
def unpack_drawing(file_handle):
    key_id, = unpack('Q', file_handle.read(8))
    country_code, = unpack('2s', file_handle.read(2))
    recognized, = unpack('b', file_handle.read(1))
    timestamp, = unpack('I', file_handle.read(4))
    n_strokes, = unpack('H', file_handle.read(2))
    image = []
    for i in range(n_strokes):
        n_points, = unpack('H', file_handle.read(2))
        fmt = str(n_points) + 'B'
        x = unpack(fmt, file_handle.read(n_points))
        y = unpack(fmt, file_handle.read(n_points))
        image.append((x, y))

    return {
        'key_id': key_id,
        'country_code': country_code,
        'recognized': recognized,
        'timestamp': timestamp,
        'image': image
    }


def unpack_drawings(filename):
    with open(filename, 'rb') as f:
        while True:
            try:
                yield unpack_drawing(f)
            except struct.error:
                break
            
# 데이터 전처리
def preprocess_data(word, data):
    drawing = data["image"]
    
    # drawing을 이미지로 변환
    image = convert_drawing_to_image(drawing)
    
    return image, word

def convert_drawing_to_image(drawing):
    # drawing을 이미지로 변환하는 로직 작성
    
    # 이미지 크기 설정
    image_height = 256
    image_width = 256
    
    # 이미지 생성
    image = np.zeros((image_height, image_width), dtype=np.uint8)
    
    # drawing의 stroke를 이미지에 그리기
    for stroke in drawing:
        x, y = stroke
        
        # stroke를 선으로 그리기
        for i in range(len(x) - 1):
            start_x = int(x[i])
            start_y = int(y[i])
            end_x = int(x[i+1])
            end_y = int(y[i+1])
            image = draw_line(image, start_x, start_y, end_x, end_y)
            
        # stroke의 점 그리기
    #    for i in range(len(x)):
    #        point_x = int(x[i])
     #       point_y = int(y[i])
     #       image = draw_point(image, point_x, point_y)

    #cv2.imshow('image',image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return image

def draw_line(image, start_x, start_y, end_x, end_y):
    # 선 그리기
    image = cv2.line(image, (start_x, start_y), (end_x, end_y), 255, 2)

    return image

def draw_point(image, point_x, point_y):
    # 점 그리기
    image[point_y, point_x] = 255
    return image

# 데이터 로드 및 전처리
def load_ndjson_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data



images = []
labels = []

#데이터 추가
file_path = "bin/anvil.bin"  # ndjson 파일의 경로
ndjson_data = unpack_drawings(file_path)

i = 0
for data in ndjson_data:
    if(i > 500):
        break
    image, label = preprocess_data(1,data)
    images.append(image)
    labels.append(label)
    i += 1
    
print(len(labels), len(images))

#데이터 추가2
file_path = "bin/full_binary_bat.bin"  # ndjson 파일의 경로
ndjson_data = unpack_drawings(file_path)

i = 0
for data in ndjson_data:
    if(i > 500):
        break
    image, label = preprocess_data(0,data)

    images.append(image)
    labels.append(label)
    i += 1
    
print(len(labels), len(images))

print(images[1001])

# 데이터셋 분할

train_images = np.array(images[:200] + images[501:800])
train_labels = np.array(labels[:200])
train_labels = np.append(train_labels, np.array(labels[501:800]))

test_images = np.array(images[200:500] + images[800:1001])
test_labels = np.array(labels[200:500])
test_labels = np.append(test_labels, np.array(labels[800:1001]))



print(len(train_images), len(train_labels))

print('모델구성중...')
# 모델 구성
model = tf.keras.Sequential([
    # 모델 구성을 위한 적절한 레이어 추가
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

print('모델 컴파일 중...')
# 모델 컴파일
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


print('레이블 변환중...')
#train_label
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)

train_labels_onehot = to_categorical(train_labels_encoded)

 #onehot_encoder = OneHotEncoder(sparse_output=False)
 #train_labels_onehot = onehot_encoder.fit_transform(train_labels_encoded.reshape(-1, 1))

#test_label
test_labels_encoded = label_encoder.fit_transform(test_labels)
test_labels_onehot = to_categorical(test_labels_encoded)
 #onehot_encoder = OneHotEncoder(sparse_output=False)
 #test_labels_onehot = onehot_encoder.fit_transform(test_labels_encoded.reshape(-1, 1))



print('모델 학습 중...')
# 모델 학습
model.fit(train_images, train_labels, epochs=4, batch_size=32, verbose=2)


print('모델 평가 중...')
# 모델 평가
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc, ' loss:',test_loss)

# 새로운 입력 데이터에 대한 예측
#new_image = preprocess_new_data(new_data)
#prediction = model.predict(np.array([new_image]))

#input_image = model.predict(test_images[200])
#input_image = np.expand_dims(input_image, axis=-1)  # 이미지에 채널 차원 추가
#input_image = np.expand_dims(input_image, axis=0)   # 배치 차원 추가

prediction = model.predict(test_images)
print('aa:',prediction)
predicted_label = np.argmax(prediction[0])
print(predicted_label)
