import sys
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-1,6,141)#input  from a file,or message
y = (x-2.5)**2-1

def dF(theta):
    return 2*(theta-2.5)

def F(theta):
    return (theta-2.5)**2-1

def main(theta,learn_rate,epsilon):
    #theta = 0.0
    #learn_rate = 0.1
    #epsilon = 1e-8
    p_history = []
    litnum = 0
    while True:
        gradient = dF(theta) 
        last_theta = theta
        theta = theta - learn_rate*gradient
        p_history.append(theta)

        litnum += 1
        if(abs(F(last_theta)-F(theta))<epsilon):
            break    
        
    plt.plot(x,y)
    plt.plot(np.array(p_history), F(np.array(p_history)), color = 'r', marker='+')
    plt.show()
    print(litnum,":",theta,F(theta))#output  to a file,or message

if __name__ == '__main__':
    theta = 0.0
    learn_rate = 0.1
    epsilon = 1e-8
    a = []
    for i in range(1, len(sys.argv)):
        a.append((str(sys.argv[i])))
    theta=float(a[0])
    learn_rate=float(a[1])
    epsilon=float(a[2])
    
    main(theta,learn_rate,epsilon)

