'''
Functions to compute spectro-temporal heterogeneity (STH) measures of grasslands clustered a priori using HDDC.
'''

import scipy as sp
from pandas import DataFrame


def compute_MDC(G):
    """Compute the mean distance to centroid MDC (or global variability, V) of the object.
    
    Input:
        -G (array): pixels values of the considered object
    
    Return:
        -MDC (float): MDC
    """
    MDC = sp.trace(sp.cov(G,rowvar=False))
    return MDC
    

def compute_W_B(G,Y):
    """Compute the intra- and the inter-classe variabilities of the clustered object.
    Input:
        -G (array): pixels values of the considered object
        -Y (vector): list of pixels' clusters
    Return:
        -W (float): intra-class variability
        -B(float): inter-class variability
    """
    B = 0
    W = 0
    
    for c in sp.unique(Y): #For each cluster present in the object
        t_c = sp.where(Y==c)[0]
        X_c = G[t_c,:]
        ni_c = t_c.size #Number of pixels associated to this cluster
        pi_c = float(ni_c)/Y.size #Proportion of class c in the object
        if ni_c == 1:
            W += 0            
        else:
            Vc = sp.cov(X_c,rowvar=False)
            W += sp.trace(Vc) * pi_c
            
        b = sp.mean(X_c,0)-sp.mean(G,0)
        b = b.reshape(b.size,1)
        Bc = sp.dot(b,b.T)
        B += sp.trace(Bc) * pi_c
    return W, B
        
def compute_E(PI):
    """Compute the entropy from soft assignment of the pixels to each cluster of the object.
    Input:
        -PI (array): belongship probability of each pixel (row) to each cluster (columns)
    Return:
        -E (float): entropy
    """
            
    E= 0
    ni,nC = PI.shape #Nbr of pixels in the object, number of clusters
    for c in range(nC): #For each cluster
        PI_c = 1./ni * sp.sum(PI[:,c]) #Average belongship probability of the obejct to this cluster
        if PI_c > sp.finfo(sp.float64).eps:
            E += - PI_c * sp.log(PI_c)
            
    return E   


def compute_4_STH_measures(X,ID,yp,T):
    """Compute the 4 STH measures (MDC, intra- and inter-class variability, entropy) of a set of objects clustered by HDDC.
    
    Input:
        - X (array): X of objects' pixels
        - ID (vector): corresponding IDs of pixels
        - yp (vector): predicted clusters of each pixel
        - T (array): belongship probabilities of each pixel to each cluster
        
    Return:
        - VAR (pandas data frame): data frame containing the 4 STH measures associated to each object
    """  
    
    list_IDs = sp.unique(ID)
    var = sp.zeros((list_IDs.size,4),'float')
    for i,id_ in enumerate(list_IDs):
        p = sp.where(ID==id_)[0]
        G = X[p,:] #Grassland's pixels
        Y = yp[p]  #Clusters of the pixels
        PI = T[p,:] #Belongship probabilities of the pixels to each cluster
        w,b = compute_W_B(G,Y)
        #Log-transformed global variability
        var[i,0] = sp.log(compute_MDC(G))
        #Log-transformed intra-class variability
        var[i,1] = sp.log(w)
        #Log-transformed inter-class variability
        var[i,2] = sp.log(b+1)
        #Entropy
        var[i,3] = compute_E(PI)
    
    #Save the 4 STH measures of all the grasslands in a pandas data frame
    VAR = DataFrame(data=var,index=list_IDs,columns=["V","W","B","E"])
    
    return VAR