# USAGE
# python gradient_descent1.py -w 8 -b 40

import argparse
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# construct the argument parse to get the arguments
def get_argument():
    ap = argparse.ArgumentParser()
    ap.add_argument("-w", type=float, default=8,
                    help="coefficient w initial ")
    ap.add_argument("-b", type=float, default=40,
                    help="coefficient b initial")
    ap.add_argument("-l", type=int, default=0,
                    help="iteration number")

    args = vars(ap.parse_args())
    return args


def plot_regression_fig(sub_ax, x, y, y_pred, iteration):
    sub_ax.set_title("Predict vs True value:{} iteration".format(iteration))
    sub_ax.set_xlabel("Area #")
    sub_ax.set_ylabel("House Value")
    sub_ax.plot(x, y_pred, "r-", label="predict line")
    sub_ax.scatter(x, y_pred, color='purple', marker='X', label="predict value")
    sub_ax.scatter(x, y, color='blue', marker='o', label='true value')
    sub_ax.legend()


def plot_fig1and2(ax, x, y, y_pred, iltnum):
    if iltnum == 1:
        plot_regression_fig(ax[0, 0], x, y, y_pred, iltnum)

    elif iltnum == 50:
        plot_regression_fig(ax[0, 1], x, y, y_pred, iltnum)


def linear_regression(w, b, loop, ax):
    data = pd.read_excel('house_data_1.xlsx')  # read house data from excel file
    # x is a list stores house area,e.g([68, 95, 102, 130, 60, 45, 30, 80, 120, 113, 150])
    x = np.array(data.iloc[:, 0].values)
    y = np.array(data.iloc[:, 1].values)  # y is a list stores house value,
    loss_value = []  # loss_value is a list stores MSE value of each learning
    m = x.size
    learn_rate = 0.0004
    iltnum = 0
    y_pred = []  # prediction function,a list stores regression value of each learning
    while True:
        y_pred = w * x + b  # a list store values of house regression price
        loss = (y_pred - y) ** 2  # loss function,a list store every y_pred(i)-y(i),i=[0:m]
        loss_value.append((0.5 * loss.sum() / m))
        grad_w = 0.5 * np.sum((y_pred - y) * x) / m  # calculate gradient for w
        grad_b = 0.5 * np.sum(y_pred - y) / m  # calculate gradient for b
        w -= learn_rate * grad_w  # from gradient decent direction to  get the next w
        b -= learn_rate * grad_b  # from gradient decent direction to  get the next b

        print("%-7d:loss %-12.3f, grad_w:%-10.3f,grad_b:%-10.3f ,(w,b):%-7.2f %-7.2f" \
              % (iltnum, loss_value[iltnum], grad_w, grad_b, w, b))
        if loop == 0:
            if iltnum > 0:
                if loss_value[iltnum] > loss_value[iltnum - 1]:  # if loss value bigger then stop
                    break
                elif round(abs(loss_value[iltnum]), 4) == round(abs(loss_value[iltnum - 1]), 4) \
                        or (round(abs(grad_w), 3) <= 0.001 and round(abs(grad_b), 2) <= 0.01):
                    print("learning end after small loss value")
                    break
        elif iltnum > loop:
            print("learning end after: %d iterations" % iltnum)
            break

        iltnum += 1
        plot_fig1and2(ax, x, y, y_pred, iltnum)
    plot_regression_fig(ax[1, 0], x, y, y_pred, iltnum)

    # plot sub-figure of loss_value list
    x = []
    for i in range(iltnum + 1):
        x.append(i)
    x = np.array(x)
    ax[1, 1].set_title("loss value curve:{} iteration".format(iltnum))
    ax[1, 1].set_xlabel("iltnum #")
    ax[1, 1].set_ylabel("loss Value")
    function = "f(x)={} *x + {}".format(round(w, 3), round(b, 3))
    ax[1, 1].plot(x, loss_value, color='purple',
                  label=function + f"\nInitial Predict Coefficient(w0,b0):({args['w']},{args['b']})" \
                        + f"\nlearn rate = {learn_rate}" \
                        + f"\nLoss value Maximum = {round(loss_value[0], 1)},Minimum = {round(loss_value[iltnum], 1)}")

    ax[1, 1].legend()


if __name__ == '__main__':
    args = get_argument()
    w = args["w"]  # 8
    b = args["b"]  # 40
    loop = args["l"]  # 0
    # create a new figure for the prediction
    plt.style.use("ggplot")
    (fig, ax) = plt.subplots(2, 2, figsize=(12, 6))
    linear_regression(w, b, loop, ax)

    plt.tight_layout()
    plt.savefig("LinearRegression_w{}_b{}.png".format(args["w"], args["b"]))
    plt.show()
    plt.close()
