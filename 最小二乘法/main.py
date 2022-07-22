import random
from tkinter import *

import cv2
import numpy as np

import PIL
from PIL import ImageTk
from lsm import LogicClass


def sp_noise(image, prob):
    '''
    添加椒盐噪声
    prob:噪声比例 
    '''
    output = np.zeros(image.shape, np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output


if __name__ == '__main__':
    root = Tk()  # 初始化Tk()

    root.title("图像去噪")  # 设置窗口标题
    root.geometry("900x500")  # 设置窗口大小 注意：是x 不是*
    root.resizable(width=False, height=False)  # 设置窗口是否可以变化长/宽，False不可变，True可变，默认为True
    myFrame = Frame(root, height=30)
    myFrame.pack()

    myLabelFrame = LabelFrame(root, text='lsm image noise filter')
    myLabelFrame.pack()
    '''
    myImage1 = Label(myLabelFrame, width=220, height=200)
    myImage1.grid(row=0, column=0)
    myLabel1 = Label(myLabelFrame, text='原始图片')
    myLabel1.grid(row=1, column=0)
'''
    myImage2 = Label(myLabelFrame, width=420, height=400)
    myImage2.grid(row=0, column=1)
    myLabel2 = Label(myLabelFrame, text='noise image')
    myLabel2.grid(row=1, column=1)

    myImage3 = Label(myLabelFrame, width=420, height=400)
    myImage3.grid(row=0, column=3)
    myLabel3 = Label(myLabelFrame, text='lsm filtered image')
    myLabel3.grid(row=1, column=3)

    '''
    image = cv2.imread("lenna.jpg", cv2.COLOR_BGR2RGB) #IMREAD_GRAYSCALE)
    image = cv2.resize(image, (300, 300))

    image1 = PIL.Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image1 = ImageTk.PhotoImage(image1)
    #myImage1.configure(image=image1)
    '''
    # 添加噪声
    #image = sp_noise(image, 0.01)
    #cv2.imwrite("lenna2.jpg",image)
    image = cv2.imread("cat2.jpg", cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (300, 300))
    image2 = PIL.Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image2 = ImageTk.PhotoImage(image2)
    myImage2.configure(image=image2)

    # 最小二乘滤波
    image = LogicClass.delete_noise(image)
    #cv2.imshow("filtered_image", image)
    image3 = PIL.Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image3 = ImageTk.PhotoImage(image3)
    myImage3.configure(image=image3)
    root.mainloop()
