import struct
from struct import unpack
import matplotlib.pyplot as plt

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


import matplotlib.pyplot as plt

def visualize_stroke(stroke):
    for path in stroke:
        print(path)
        x, y = path
        plt.plot(x, y, marker='.')
    plt.gca().invert_yaxis()
    plt.show()
    

i = 0
for drawing in unpack_drawings('anvil.bin'):
    if(i >= 1):
        break
    print(drawing['image'][0])
    #visualize_stroke(drawing['image']) #시각화
    i = i+1
