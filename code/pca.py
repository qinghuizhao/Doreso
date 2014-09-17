
from numpy import *
def pca(data,nRedDim=0,normalise=1):
   
    # 
    m = mean(data,axis=0)
    data -= m
    #
    C = cov(transpose(data))
     
    evals,evecs = linalg.eig(C)
    indices = argsort(evals)
    indices = indices[::-1]
    evecs = evecs[:,indices]
    evals = evals[indices]
    if nRedDim>0:
        evecs = evecs[:,:nRedDim]
   
    if normalise:
        for i in range(shape(evecs)[1]):
            evecs[:,i] / linalg.norm(evecs[:,i]) * sqrt(evals[i])
    
    x = dot(transpose(evecs),transpose(data))
    y=transpose(dot(evecs,x))+m
    return x,y,evals,evecs
