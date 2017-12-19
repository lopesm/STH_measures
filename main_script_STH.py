'''
Script to compute 4 spectro-temporal heterogeneity measures of a set of grasslands belonging to the same landscape.
Compare them to a biodiversity index computed from the field.
'''
import argparse
import scipy as sp
from pandas.stats.api import ols
#from functions import get_samples_from_roi
from hdda import HDGMM
from compute_sth import compute_4_STH_measures


###Parser initialization
parser = argparse.ArgumentParser(description="Run script for computing heterogeneity measures of grasslands from their spectral clustering.")
parser.add_argument("-X",action='store',dest='X',default='X_ms_smoothed_d2l10e4_grasslands_indices.npy', help='X of grasslands pixels',type=str)
parser.add_argument("-ID",action='store',dest='ID',default='ID_grasslands_indices.npy', help='ID of grasslands pixels',type=str)
parser.add_argument('-Cmin', action='store', dest='Cmin', default=2, help='Lowest number of clusters',type=int)
parser.add_argument('-Cmax', action='store', dest='Cmax', default=20, help='Highest number of clusters',type=int)
parser.add_argument('-INT', action='store', dest='INT', default=2, help='Interval for the clusters range',type=int)
parser.add_argument('-M', action='store', dest='M', default="M2", help='HDDC model name',type=str)
parser.add_argument('-th', action='store', dest='th', default=0.1, help='Threshold parameter',type=float)
parser.add_argument('-CRIT', action='store', dest='CRIT', default='icl', help='Model selection parameter',type=str)
parser.add_argument('-n_init', action='store', dest='n_init', default=10, help='Number of model initializations',type=int)
parser.add_argument('-n_jobs', action='store', dest='n_jobs', default=2, help='Number of jobs to run in parallel',type=int)
args = parser.parse_args()
    
### Load data (X and ID of grasslands -ID is necessary to work at the object level-)
X = sp.load(args.X)
ID = sp.load(args.ID)
#These files were obtained from the following function:
#raster_name = 'serie_spot5_coteaux_concat_smoothed_d2l10e4.tif'
#mask_grasslands_ID = 'prairies_indices_id.tif'
#X, ID = get_samples_from_roi(raster_name,mask_grasslands_ID)

#Field data
data = sp.loadtxt('biodiv_indices.csv',delimiter=",",skiprows=1)
#Shannon index of the grasslands (ordered according to increasing ID)
H = data[:,3] 

###Clustering parameters
M = args.M
th = args.th
criteria = args.CRIT
n_init = args.n_init
n_jobs = args.n_jobs

### Run HDDC clustering on all the pixels 
#Number of clusters
CLUSTERS = range(args.Cmin,args.Cmax+1,args.INT)

for C in CLUSTERS:    
    model = HDGMM()
    #Run 10 model initializations and returns the best model in terms of ICL
    model.fit_best_init(X,C=C,n_init=n_init,M=M,th=th,criteria=criteria,n_jobs=n_jobs)
    #pickle.dump(model,open(filenaname+"_M2_"+str(C)+"C.p","wb")) #(model can be saved to prevent from re-running)
    #Predicts the pixels clusters
    yp=model.predict(X)
    T = model.T
    
    ### Compute the 4 STH measures of each grassland listed by their ID
    VAR = compute_4_STH_measures(X,ID,yp,T)
    
    #Compute the adjusted R2 between the STH measures and the biodiversity index (examples)
    R2_1 = ols(y=H, x=VAR['W']).r2_adj
    R2_2 = ols(y=H, x=VAR[['V','W','B','E']]).r2_adj
    print R2_1
    print R2_2
 
