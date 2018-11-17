#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 12:36:05 2018

@author: shujaat
"""

class Linear_regressor:
    
    
     # Calculate Cost/error function of our reggression line
    def cost_function(self,m,b,points):
        N = float(len(points))
        c=0
        for i in range(0,len(points)):
            x=points[i,0]
            y=points[i,1]
            c += (((m*x)+b)-y)**2
            return c/N
    
 #Run gradient descent for one epoch with data points   
    def gradient_step(self,b,m,a,points):
        b_gradient=0.0
        m_gradient=0.0
        N=float(len(points))
        for i in range(0,len(points)):
            x=points[i,0]
            y=points[i,1]
            b_gradient += -(2/N) * (y - ((m * x) + b))
            m_gradient += -(2/N) * x * (y - ((m * x) + b))
            new_b=b-(a*b_gradient)
            new_m=m-(a*m_gradient)
            return [new_b,new_m]
        
        
# Gradient_runner for iterations
    def gradient_runner(self,no_of_iterations,points,b,m,a):
        for i in range(no_of_iterations):
            b,m=self.gradient_step(b,m,a,points)
           
        return [b,m]
        
    
    
# predict after training  
    def predict(self,m,b,points):
        output=[]
        
        for i in range(0,len(points)):
            x=points[i,0]
            y=(m*x)+b
            output.append(y)
            
        return output
import numpy as np
points = np.genfromtxt("train.csv", delimiter=",")
test = Linear_regressor()
print(test.cost_function(0,1,points))
bn,mn=test.gradient_runner(1,points,0,1,0.01)
print(test.cost_function(mn,bn,points))


