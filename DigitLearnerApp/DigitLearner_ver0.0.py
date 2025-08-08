import sklearn.datasets
import sklearn.svm
import PIL.Image
import numpy as np

# 画像ファイルを数値リストに変換
def imageToData(filename):
    grayImage = PIL.Image.open(filename).convert("L")
    grayImage = grayImage.resize((8,8),PIL.Image.Resampling.LANCZOS)

    npImage = np.asarray(grayImage, dtype = float)
    npImage = 16 - np.floor(17 * npImage / 256)
    npImage = npImage.flatten()

    return npImage

#数字を予測
def predictDigits(data):
    digits = sklearn.datasets.load_digits()
    
    clf = sklearn.svm.SVC(gamma = 0.001)
    clf.fit(digits.data, digits.target)

    n = clf.predict([data])
    print("予測=",n)

# 画像ファイルを数値リストに変換
data = imageToData("9.png")

# 数字を予測
predictDigits(data)
