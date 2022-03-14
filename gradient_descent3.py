# USAGE
# python gradient_descent3.py -w1 8 -w2 12 -w3 -20 -b -100

import argparse
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-w1", type=float, default=8,
                help="coefficient w1 initial ")
ap.add_argument("-w2", type=float, default=12,
                help="coefficient w1 initial")
ap.add_argument("-w3", type=float, default=-20,
                help="coefficient w1 initial")
ap.add_argument("-b", type=float, default=-100,
                help="coefficient b initial")

args = vars(ap.parse_args())
w1 = args["w1"]  # 8,10,10264
w2 = args["w2"]  # 12,15, 49350
w3 = args["w3"]  # 12,15, 49350
b = args["b"]  # -1,-10, -100533.75

df1 = pd.read_excel('house_value.xlsx')  # read first sheet of xlsx
# x1 = np.array([68, 95, 102, 130, 60, 45, 30, 80, 120, 113, 150])
x1 = np.array(df1.iloc[:, 0].values)  # house area
x2 = np.array(df1.iloc[:, 1].values / 1000)  # average income
x3 = np.array(df1.iloc[:, 2].values)  # house years
# y = np.array([714.592, 956.877, 1153.582, 1293.667, 600.000, 520.000, 280.000, 845.000, 1150.000, 1120.000, 1490.234])
y = np.array(df1.iloc[:, 3].values / 1000)  #house price

m = x1.size

learn_rate1 = 0.0001
learn_rate2 = 0.0004
iltnum = 0
loss_value = []

# create a new figure for the prediction
plt.style.use("ggplot")
(fig, ax) = plt.subplots(2, 2, figsize=(12, 8))

# for i in range(100):
while True:
    y_pred = w1 * x1 + w2 * x2 + w3 * x3 + b  # function,linear regression
    loss = (y_pred - y) ** 2  # loss function ,a list store every y_pred(i)-y(i) loss value,i=[0:m]
    loss_value.append((0.5 * loss.sum() / m))  # MSE ,all the(y_pred(i)-y(i)) loss values summary
    grad_w1 = 0.5 * np.sum((y_pred - y) * x1) / m  # partial derivative, gradient for w1
    grad_w2 = 0.5 * np.sum((y_pred - y) * x2) / m  # partial derivative, gradient for w2
    grad_w3 = 0.5 * np.sum((y_pred - y) * x3) / m  # partial derivative, gradient for w3
    grad_b = 0.5 * np.sum(y_pred - y) / m  # partial derivative, gradient for b
    w1 -= learn_rate1 * grad_w1  # gradient descent for w1 , from gradient decent direction to  get the next w1
    w2 -= learn_rate2 * grad_w2  # gradient descent for w2 , from gradient decent direction to  get the next w2
    w3 -= learn_rate2 * grad_w3  # gradient descent for w2 , from gradient decent direction to  get the next w3
    b -= learn_rate2 * grad_b    # gradient descent for b , from gradient decent direction to  get the next b

    print(
        "%-7d:loss %-6.3f, grad_w1:%-8.4f,grad_w2:%-8.4f,grad_w3:%-8.4f,grad_b:%-8.5f ,(w1,w2,w3,b):%-5.2f %-5.2f "
        "%-5.2f %-5.2f" \
        % (iltnum, loss_value[iltnum], grad_w1, grad_w2, grad_w3, grad_b, w1, w2, w3, b))
    if iltnum == 0:
        ax[0, 0].set_title("Predict vs True value:{} iteration".format(iltnum))
        ax[0, 0].set_xlabel("Area #")
        ax[0, 0].set_ylabel("House Value")
        ax[0, 0].scatter(x1, y_pred, color='red', marker='X', label="predict value")
        ax[0, 0].scatter(x1, y, color='blue', marker='o', label='true value')
        ax[0, 0].legend()
    elif iltnum == 50:
        ax[0, 1].set_title("Predict vs True value:{} iteration".format(iltnum))
        ax[0, 1].set_xlabel("Area #")
        ax[0, 1].set_ylabel("House Value")
        ax[0, 1].scatter(x1, y_pred, color='red', marker='X', label="predict value")
        ax[0, 1].scatter(x1, y, color='blue', marker='o', label='true value')
        ax[0, 1].legend()
    if iltnum > 0:
        if loss_value[iltnum] > loss_value[iltnum - 1]:  # if loss value becomes bigger then stop
            break
        elif round(abs(loss_value[iltnum]), 4) == round(abs(loss_value[iltnum - 1]), 4) \
                or (round(abs(grad_w1), 3) <= 0.001 and round(abs(grad_b), 2) <= 0.01):
            break
    iltnum += 1
print(w1, w2, w3, b)

ax[1, 0].set_title("Predict vs True value:{} iteration".format(iltnum))
ax[1, 0].set_xlabel("Area #")
ax[1, 0].set_ylabel("House Value")
ax[1, 0].scatter(x1, y_pred, color='red', marker='X', label="predict value")
ax[1, 0].scatter(x1, y, color='blue', marker='o', label='true value')
ax[1, 0].legend()

x = []
for i in range(iltnum + 1):
    x.append(i)
x = np.array(x)
ax[1, 1].set_title("loss value curve:{} iteration".format(iltnum))
ax[1, 1].set_xlabel("iltnum #")
ax[1, 1].set_ylabel("loss Value")
function = "f(x1,x2,x3)={} *x1 + {} *x2 + {} *x3 + {}".format(round(w1, 3), round(w2, 3),  round(w3, 3), round(b, 3))
ax[1, 1].plot(x, loss_value, color='purple',
              label=function + f"\nInitial Predict Coefficient (w1o,w2o,w3o,b0):({args['w1']},{args['w2']},{args['w3']},{args['b']})" \
                    + f"\nlearn rate = {learn_rate1}" \
                    + f"\nLoss value Maximum = {round(loss_value[0], 1)},Minimum = {round(loss_value[iltnum], 1)}")
# plt.ylim(331.805, 331.826)
ax[1, 1].legend()
plt.tight_layout()
plt.savefig("LinearRegression_w1{}_w2{}_b{}.png".format(args["w1"], args["w2"], args["b"]))
plt.show()
plt.close()
