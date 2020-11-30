import cv2
import numpy as np
import os
import pandas as pd
from skimage.feature import greycomatrix, greycoprops
from skimage import io, color, img_as_ubyte


# 数据划分为5个等级
def fivegrade(data):
    grade = 0
    if 0 < data < 0.2:
        grade = 0
    elif 0.2 <= data < 0.4:
        grade = 1
    elif 0.4 <= data < 0.6:
        grade = 2
    elif 0.6 <= data < 0.8:
        grade = 3
    else:
        grade = 4
    return grade


# 求三阶矩
def val(x=None):
    mid = np.mean(((x - x.mean()) ** 3))
    return np.sign(mid) * abs(mid) ** (1/3)


def getColourData(img):
    data = [0] * 9
    r, g, b = cv2.split(img)
    rd = np.asarray(r) / 255
    gd = np.asarray(g) / 255
    bd = np.asarray(b) / 255
    data[0] = rd.mean()
    data[1] = gd.mean()
    data[2] = bd.mean()
    data[3] = rd.std()
    data[4] = gd.std()
    data[5] = bd.std()
    data[6] = val(rd)
    data[7] = val(gd)
    data[8] = val(bd)
    rsj = fivegrade(data[6])
    gej = fivegrade(data[4])
    gsj = fivegrade(data[7])
    coInfo = [rsj, gej, gsj]
    return coInfo


def getTexture(img):
    gray = color.rgb2gray(img)
    image = img_as_ubyte(gray)
    bins = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255])
    inds = np.digitize(image, bins)
    matrix_coocurrence = greycomatrix(inds, [2], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=256)
    contrast = greycoprops(matrix_coocurrence, 'contrast')
    asm = greycoprops(matrix_coocurrence, 'ASM')
    con_m, asm_m = np.mean(contrast), np.mean(asm)
    con_z, asm_z = int(round(con_m)), int(round(asm_m))
    return [con_z, asm_z]


def createcsv(path):
    labels = []
    for image in os.listdir(path):
        name = path + '\\' + image
        img = cv2.imread(name)
        # cv2.imshow('img', img)

        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        H = img_hsv[..., 0]
        S = img_hsv[..., 1]
        V = img_hsv[..., 2]
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        L = img_lab[..., 0]
        A = img_lab[..., 1]
        B = img_lab[..., 2]

        img_H = cv2.GaussianBlur(H, (3, 3), 0)
        img_A = cv2.GaussianBlur(A, (3, 3), 0)
        img_B = cv2.GaussianBlur(B, (3, 3), 0)
        reth, threshh = cv2.threshold(img_H, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        reta, thresha = cv2.threshold(img_A, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        retb, threshb = cv2.threshold(img_B, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img_hab = cv2.merge((threshh, thresha, threshb))
        # cv2.imshow('hab', img_hab)
        finalimg = cv2.bitwise_and(img_hab, img)
        # cv2.imshow('final', finalimg)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        colorInf = getColourData(finalimg)  # 计算颜色特征：r三阶矩，g二阶矩，g三阶矩
        textureInf = getTexture(finalimg)  # 计算纹理特征：对比度，asm
        allInfo = colorInf + textureInf
        labels.append((allInfo[0], allInfo[1], allInfo[2], allInfo[3], allInfo[4], image))
    labels = pd.DataFrame(labels)
    labels.to_csv('labels.txt', header=None, index=None)
