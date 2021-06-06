# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 12:14:48 2021

@author: tmdal
"""

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-poster')

f=lambda x,y: (3*y)/(2*(x+1))+np.sqrt(x+1)
h=1.0
x=np.arange(0,10+h,h)
y0=0

y=np.zeros(len(x))
y[0]=y0

for i in range(0,len(x)-1):
    y[i+1]=y[i]+h*f(x[i],y[i])
    
plt.figure(figsize=(12,8))
plt.plot(x,y,'b--',label='Approximate')
plt.plot(x,(np.sqrt(x+1)**3)*np.log(x+1),'g',label='Exact')
plt.title('Approximate and Exact Soution\
for SImple ODE')
plt.xlabel('x')
plt.ylabel('y(x)')
plt.grid()
plt.legend(loc='lower right')
plt.show