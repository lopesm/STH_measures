'''
Script to compute 4 spectro-temporal heterogeneity measures of a set of grasslands belonging to the same landscape.
Compare them to a biodiversity index computed from the field.
'''
import scipy as sp
from pandas import DataFrame
from pandas.stats.api import ols
from functions import get_samples_from_roi
from hdda import HDGMM
from compute_sth import *

### Load data (X and ID of grasslands -ID is necessary to work at the object level-)
raster_name = '/media/mlopes/ntfs/SPOT5-Take5/coteaux/serie_spot5_coteaux_concat_smoothed_d2l10e4.tif'
mask_grasslands_ID = '/home/mlopes/Image_Data/Bota_modele_GB/sig/prairies_indices_id.tif'
#X, ID = get_samples_from_roi(raster_name,mask_grasslands_ID)
#OR (already saved in npy format)
X = sp.load('X_ms_smoothed_d2l10e4_grasslands_indices.npy')
ID = sp.load('ID_grasslands_indices.npy')

#Field data
data = sp.loadtxt('biodiv_indices.csv',delimiter=",",skiprows=1)
#Shannon index of the grasslands (ordered according to increasing ID)
H = data[:,3] 

### Run HDDC clustering on all the pixels 
#Number of clusters
CLUSTERS = range(2,20)

for C in CLUSTERS:
    #Model name
    M = "M2"
    #Threshold parameter
    th = 0.1
    #Model selection criteria
    criteria = "icl"
    #Number of initializations
    n_init = 10
    #Number of jobs to run in parallel
    n_jobs=  2
    
    model = HDGMM()
    #Run 10 model initializations and returns the best model in terms of ICL
    model.fit_best_init(X,C=C,n_init=n_init,M=M,th=th,criteria=criteria,n_jobs=n_jobs)
    #pickle.dump(model,open(filenaname+"_M2_"+str(C)+"C.p","wb")) #(model can be saved to prevent from re-running)
    #Predicts the pixels clusters
    yp=model.predict(X)
    
    
    ### Compute the 4 STH measures of each grassland listed by their ID
    list_IDs = sp.unique(ID)
    var = sp.zeros((list_IDs.size,4),'float')
    for i,id_ in enumerate(list_IDs):
        p = sp.where(ID==id_)[0]
        G = X[p,:] #Grassland's pixels
        Y = yp[p]  #Clusters of the pixels
        PI = model.T[p,:] #Belongship probabilities of the pixels to each cluster
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
    
    #Compute the adjusted R2 between the STH measures and the biodiversity index (examples)
    R2_1 = ols(y=H, x=VAR['W']).r2_adj
    R2_2 = ols(y=H, x=VAR[['V','W','B','E']]).r2_adj
    print R2_1
    print R2_2
 
