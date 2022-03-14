# python3 gradient_descent1.py -w 8 -b 20
import numpy as np
import argparse
from matplotlib import pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("-w", type=float, default=8,
                help="coefficient w initial ")
ap.add_argument("-b", type=float, default=20,
                help="coefficient b initial")

args = vars(ap.parse_args())
w = args["w"]  # 8,10,10264
b = args["b"]  # -1,-10, -100533.75

m = 11
x = np.array([68, 95, 102, 130, 60, 45, 30, 80, 120, 113, 150])
y = np.array([714.592, 956.877, 1153.582, 1293.667, 600.000, 520.000, 280.000, 845.000, 1150.000, 1120.000, 1490.234])
# y is house value, unit is kRMB

learningRate = 0.0004
iltnum = 0
loss_value = []

# create a new figure for the prediction
plt.style.use("ggplot")
(fig, ax) = plt.subplots(2, 2, figsize=(12, 8))

while True:
    y_pred = w * x + b  # function,linear regression
    loss = (y_pred - y) ** 2  # loss function ,is a list,store every y_pred(i)-y(i) loss value,i=[0:m]
    loss_value.append((0.5 * loss.sum() / m))  # sum all MSEs and store it in a list for each iteration
    grad_w = 0.5 * np.sum((y_pred - y) * x) / m  # partial derivative, calculate gradient for w
    grad_b = 0.5 * np.sum(y_pred - y) / m  # partial derivative, calculate gradient for b
    w -= learningRate * grad_w  # follow gradient descent direction to  get the next w
    b -= learningRate * grad_b  # follow gradient descent direction to  get the next b

    print("%-7d:loss %-6.3f, grad_w:%-10.4f,grad_b:%-10.5f ,(w,b):%-7.2f %-7.2f" \
          % (iltnum, loss_value[iltnum], grad_w, grad_b, w, b))
    if iltnum == 0:
        ax[0, 0].set_title("Predict vs True value:{} iteration".format(iltnum))
        ax[0, 0].set_xlabel("Area #")
        ax[0, 0].set_ylabel("House Value")
        ax[0, 0].plot(x, y_pred, 'r-', label="predict value")
        ax[0, 0].scatter(x, y, color='blue', marker='o', label='true value')
        ax[0, 0].legend()
    elif iltnum == 50:
        ax[0, 1].set_title("Predict vs True value:{} iteration".format(iltnum))
        ax[0, 1].set_xlabel("Area #")
        ax[0, 1].set_ylabel("House Value")
        ax[0, 1].plot(x, y_pred, 'r-', label="predict value")
        ax[0, 1].scatter(x, y, color='blue', marker='o', label='true value')
        ax[0, 1].legend()
    if iltnum > 0:
        if loss_value[iltnum] > loss_value[iltnum - 1]:  # if loss value becomes bigger then stop
            break
        elif round(loss_value[iltnum], 4) == round(loss_value[iltnum - 1], 4) \
                or (round(abs(grad_w), 3) <= 0.001 and round(abs(grad_b), 2) <= 0.01):
            break
    iltnum += 1

ax[1, 0].set_title("Predict vs True value:{} iteration".format(iltnum))
ax[1, 0].set_xlabel("Area #")
ax[1, 0].set_ylabel("House Value")
ax[1, 0].plot(x, y_pred, 'r-', label="predict value")
ax[1, 0].scatter(x, y_pred, color='green', marker='X', label="predict value")
ax[1, 0].scatter(x, y, color='blue', marker='o', label='true value')
ax[1, 0].legend()

x = []
for i in range(iltnum + 1):
    x.append(i)
x = np.array(x)
ax[1, 1].set_title("loss value curve:{} iteration".format(iltnum))
ax[1, 1].set_xlabel("iltnum #")
ax[1, 1].set_ylabel("loss Value")
function = "f(x)={} *x + {}".format(round(w, 3), round(b, 3))
ax[1, 1].plot(x, loss_value, color='purple',
              label=function + f"\nInitial Predict Coefficient (w0,b0):({args['w']},{args['b']})" \
                    + f"\nlearn rate = {learningRate}" \
                    + f"\nLoss value Maximum = {round(loss_value[0], 1)},Minimum = {round(loss_value[iltnum], 1)}")
# plt.ylim(331.805, 331.826)
ax[1, 1].legend()
plt.tight_layout()
plt.savefig("LinearRegression_w{}_b{}.png".format(args["w"], args["b"]))
plt.show()
plt.close()

print(w, b)
