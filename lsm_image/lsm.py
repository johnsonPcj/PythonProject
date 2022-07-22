import numpy as np
from scipy.optimize import leastsq


class LogicClass:
    def fun(w, x):  # linear_regression
        f = np.poly1d(w)
        return f(x)
    @staticmethod
    def error(w, x, y):  # loss-function
        regularization = 0.01
        ret = LogicClass.fun(w, x) - y
        ret = np.append(ret, np.sqrt(regularization) * w)
        return ret

    @staticmethod
    def delete_noise(image):
        X = np.arange(0, 9)
        mask = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        for j in range(1, image.shape[0] - 1):
            for i in range(1, image.shape[1] - 1):
                for h in range(3):
                    for g in range(3):
                        y = j + h - 1
                        x = i + g - 1
                        mask[h * 3 + g] = image[x, y]
                p0 = np.random.randn(2)
                para = leastsq(LogicClass.error, p0, args=(X, mask))
                k, b = para[0]
                value = []
                for n in range(9):
                    value.append(mask[n] - (k * n + b))
                if np.argmax(np.absolute(value)) == 4:
                    image[i, j] = k * 4 + b
        return image
