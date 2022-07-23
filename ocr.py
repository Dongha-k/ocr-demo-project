from re import I
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob


img = cv2.imread('./image/digits.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]
x = np.array(cells)
train = x[:, :].reshape(-1, 400).astype(np.float32)

k = np.arange(10)
train_labels = np.repeat(k, 500)[:, np.newaxis]
np.savez('trained.npz', train = train, train_labels = train_labels)

FILE_NAME = 'trained.npz'
# 파일로부터 학습 데이터를 불러옵니다.
def load_train_data(file_name):
    with np.load(file_name) as data:
        train = data['train']
        train_labels = data['train_labels']
    return train, train_labels

# 손 글씨 이미지를 (20 x 20) 크기로 Scaling합니다.
def resize20(image):
    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    bg_img = np.zeros((20, 20), np.uint8)

    h = len(gray)
    w = len(gray[0])
    h, w = int(20 * (h / max(h, w))), int(20 * (w / max(h, w)))
    if h % 2 == 1:
        h += 1
    elif w % 2 == 1:
        w += 1
    
    gray_resize = cv2.resize(gray, (w, h))
    

    bg_img[(10 - h // 2) : (10 + h // 2), (10 - w // 2) : (10 + w // 2)] = gray_resize

    # plt.imshow(cv2.cvtColor(gray_resize, cv2.COLOR_GRAY2RGB))
    # plt.show()

    cv2.imshow('Image', bg_img)
    cv2.waitKey(0)

    # 최종적으로는 (1 x 400) 크기로 반환합니다.
    return bg_img.reshape(-1, 400).astype(np.float32)

def check(test, train, train_labels):
    knn = cv2.ml.KNearest_create()
    knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)
    # 가장 가까운 5개의 글자를 찾아, 어떤 숫자에 해당하는지 찾습니다.
    ret, result, neighbours, dist = knn.findNearest(test, k=5)
    return result


train, train_labels = load_train_data(FILE_NAME)




image = cv2.imread('test.jpg')
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(image_gray, 127, 255, 0)
contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

cv2.imwrite('result.png', image)
i = 0

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    num = thresh[y : y + h, x: x + w]
    cv2.imwrite('./result/result_' + str(i) + '.png', num)
    
    i += 1

predict = []
for n in range(i):
    test = resize20(glob.glob('./result/result_{}.png'.format(str(n)))[0])
    result = check(test, train, train_labels)
    predict.append(result)


j = 0
font =  cv2.FONT_HERSHEY_PLAIN
blue = (0, 255, 0)

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)

    img = cv2.putText(image, str(int(predict[j][0])), (x, y), font, 2, blue, 1, cv2.LINE_AA)
    j += 1

plt.imshow(image)
plt.show()
cv2.imwrite('final_result.png', image)