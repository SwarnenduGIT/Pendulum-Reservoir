# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 12:32:35 2021

@author: swarn
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import multiprocessing as mp
import time


def lorenz(arr,t):
    x = arr[0]
    y = arr[1]
    z = arr[2]
    
    dx = np.zeros(3)
    
    rho = 28.0
    
    dx[0] = 10*(y - x)
    dx[1] = -x*z + rho*x - y
    dx[2] = x*y - (8/3)*z
    return dx


def sim_pend(xyz,t,l,w,F):
    x = xyz[0]
    y = xyz[1]
    
    dx = np.zeros(2)
    g = 10
    k = 0.05
        
    dx[0] = y
    dx[1] = - k*y - (g/l)*np.sin(x) + F*np.sign(np.sin(w*t))

    return dx



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
    
    # Preparing Lorenz system data
    x0 = np.random.random(3)
    t = np.linspace(0,1200,12001)
    x = odeint(lorenz,x0,t)
    
    
    tr_len = 4100       # training length
    test_len = 1100     # testing length
    
    ran = tr_len + test_len
    
    tran = 1000
    u_sr = x[tran:tran+ran,0]    # x - state variable in input
    v_sr = x[tran:tran+ran,2]    # z - state variable in output
    
    mn = np.min(u_sr)
    mx = np.max(u_sr)
    u = (u_sr-mn)/(mx-mn)       #scaling the input in range [0:1]
    
    N = 10      #number of smapling cycle
    kt = 20     #number of sampled data per cycle
    
    F = 1 + u       #forcing amplitude (input scaled in range [1:2])
    w = 1.0         #forcing frequency
    l = 1.0         #length of the pendulum
    
    
    print('Generating reservoir states, please wait...\n It may take few minutes.')    
    
    x0 = np.array([0,0])
    pool = mp.Pool(mp.cpu_count()-2)
    out = pool.starmap(integ,[(N,kt,l,w,f) for f in F])
    pool.close()
    pool.join()
    
    temp = np.array(out)
    x_st = np.copy(temp)
    
    
    #Creating reservoir states vectors for regression
    
    mem = 100       # memory (Number of previous reservoir dynamics to consider)
    wei = np.arange(1,0,-1/mem)
    for i in range(1,mem):
        x_st = np.hstack((wei[i]*np.roll(temp,i,axis=0),x_st))
        
    x_st = x_st[mem:,:]


    xs_tr = x_st[:tr_len,:]         #Reservoir states corresponding to training input
    xs_test = x_st[tr_len:,:]       #Reservoir states corresponding to test input
    
    y_tr = v_sr[mem:mem+tr_len]     # Training labels
    y_test = v_sr[mem+tr_len:]      # Target dynamics
      
    w_ij = np.dot(np.linalg.pinv(xs_tr),y_tr)   #Regression
    f_out = np.dot(xs_test,w_ij)                #Predicted dynamics
    
    err = sum((y_test - f_out)**2)/sum((y_test)**2)
       
    print('\n Relative squared error: ',err)
    
    req_time = time.time() - start_time
    print('\nCalculation time: ',req_time)
    
    
    plt.plot(f_out)
    plt.plot(y_test)
    plt.show()