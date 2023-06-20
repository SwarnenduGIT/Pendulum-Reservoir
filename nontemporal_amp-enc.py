# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 16:50:10 2020

@author: swarn
"""



import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import multiprocessing as mp
import time

def sim_pend(xyz,t,w,l,F):
    x = xyz[0]
    y = xyz[1]
    
    dx = np.zeros(2)
    g = 10.0
    k = 0.05
        
    dx[0] = y
    dx[1] = - k*y - (g/l)*np.sin(x) + F*np.sign(np.sin(w*t))

    return dx


def poly(x):
    y = (x-3)*(x-2)*(x-1)*x*(x+1)*(x+2)*(x+3)
    return y



def integ(N,kt,l,w,F):
    om = w    # Data sampling cycle frequency
    tau = 2*np.pi/(om*kt) # Sampling interval
    
    t = np.arange(0,N*kt*tau,tau) # time points
    x0 = np.array([0,0])
    x = odeint(sim_pend,x0,t,args=(l,w,F,))
    
    eps = 0  # Noise strength
    noise = eps*(2*np.random.random(len(t))-1)
    
    return x[:,0]+noise




if __name__ == '__main__':

    start_time = time.time()    
    
    tr_len = 500       # training length
    test_len = 100     # testing length
    
    ran = tr_len + test_len
    
    x_sr = np.linspace(-3,3,ran)
    np.random.shuffle(x_sr)
    y_sr = poly(x_sr)

    u = (x_sr+3)/6  #scaling the input in range [0:1]
    
    N = 10      #number of smapling cycle
    kt = 20     #number of sampled data per cycle
    
    F = 1 + u       #forcing amplitude (input scaled in range [1:2])
    w = 1.0         #forcing frequency
    l = 1.0         #length of the pendulum
    
    
    avg = 10    #Number of runs to be averaged upon
    err_st = []    
    for i in range(avg):
        print('Averaging Calculation progress; %d out of %d.'%(i+1,avg),end='\r')
        #Generating reservoir states 
        pool = mp.Pool(mp.cpu_count()-2)
        out = pool.starmap(integ,[(N,kt,l,w,f) for f in F])
        pool.close()
        pool.join()
        
        x_st = np.array(out)
    
        xs_tr = x_st[:tr_len,:]     #Reservoir states corresponding to training input
        xs_test = x_st[tr_len:,:]   #Reservoir states corresponding to test input
        
        y_tr = y_sr[:tr_len]        # Training labels
        x_test = x_sr[tr_len:]      # Test input
        y_test = y_sr[tr_len:]      # Target output
          
        w_ij = np.dot(np.linalg.pinv(xs_tr),y_tr)   # Regression
        f_out = np.dot(xs_test,w_ij)                # Prediction
        
        err = sum((y_test - f_out)**2)/sum((y_test)**2)
            
        err_st.append(err)
    
    print('\nAvg. relative squared error: ',sum(err_st)/avg)
    
    
    req_time = time.time() - start_time
    print('\nCalculation time: ',req_time)
    
    
    

    # plotting the result for last averaging iteration
    plt.plot(np.arange(-3,3,0.01),poly(np.arange(-3,3,0.01)),label = 'Actual')
    plt.scatter(x_test,f_out,s=1.0,label = 'Predicted',c='red')
    plt.text(1,50,'Accuracy = %1.8f'%err)
    plt.legend()
    plt.show()
    
