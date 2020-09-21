import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

xdf=pd.read_csv(r"linearX.csv")
ydf=pd.read_csv(r"linearY.csv")

x=np.array(xdf)
y=np.array(ydf)

plt.scatter(xdf,ydf,color='green')
plt.xlabel("Acidity of wine")
plt.ylabel("Density of Wine")
plt.show()

from sklearn.model_selection import train_test_split
X,x_test,y,y_test=train_test_split(xdf,ydf,test_size=0.2)
y_training=y

xmean=np.mean(X)
xvar=np.var(X)
x=(X-xmean)/xvar
ones=np.ones([x.shape[0],1])
x=np.concatenate([ones,x],1)

def adaptive_lr(x,y,theta,lr,gamma):
    costs=[]
    m=len(y)
    theta_his=[]
    cnt=0
    j_old=0
    lr_a = []
    lr_a.append(lr)
    e=np.dot(x,theta)-y
    j_cur=np.sum(e**2)/(2*m)
   # pred.append(np.dot(x,theta))
    costs.append(j_cur)
    theta_his.append(theta)
    cnt+=1
    while abs(j_cur-j_old)>gamma:
        j_old=j_cur
        n_lr = lr_a[-1]/np.sqrt(cnt)
        lr_a.append(n_lr)
        grad=x.T.dot(e)/m
        theta=theta-lr_a[-1]*grad
        theta_his.append(theta)
        
        e=np.dot(x,theta)-y
        j_cur=np.sum(e**2)/(2*m)
        costs.append(j_cur)
        cnt+=1
        
    return theta_his,costs,cnt,lr_a 

lr =1
max_iter=2000
gamma=0.000000000001

y = np.array(y).flatten()
theta=([0,0])
theta_hist,costs,num,lr_a=adaptive_lr(x,y,theta,lr,gamma)
theta=theta_hist[-1]

plt.title('Cost Function J')
plt.xlabel('No. of iterations')
plt.ylabel('Cost')
plt.plot(costs)
plt.show()

plt.plot(X, y_training,'*',label='Data points', markersize=6)
plt.xlabel('Acidity of wine')
plt.ylabel('Density of wine')
plt.title('Linear Regression')
plt.plot(X,np.dot(x, theta),label='prediction',color='red')
plt.legend()
plt.show(block=False)

x_test=np.array(x_test)
xp=x_test[5]
y_pred=(theta[0]+theta[1]*xp)
print("actual value:{0} and Predicted value:{1}".format(y[5],y_pred))

def error(x,y,theta):
    return np.sum((x.dot(theta)-y)**2)/(2*y.size)

x_=np.linspace(-0.6,2,30) #x axis range
y_=np.linspace(-1,1,30) #y axis range
x_,y_=np.meshgrid(x_,y_)
zs=np.array([error(x,y,theta) for theta in zip(np.ravel(x_), np.ravel(y_))]) 
zs=zs.reshape(x_.shape) 

fig = plt.figure()
CS = plt.contour(x_,y_,zs) #plotting contour
plt.plot([theta[0]], [theta[1]], color='r', marker='o', label='Optimal Value') #optimum value
plt.legend() 
plt.title('Contour plot')
plt.show()

