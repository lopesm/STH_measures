# -*- coding: utf-8 -*-
"""
Script containing basic functions.
"""

import scipy as sp
from osgeo import gdal


def get_samples_from_roi(raster_name,roi_name,datatype="uint32"):
    '''
    The function get the set of pixels given the thematic map. Both map should be of same size. Data is read per block.
    Input:
        raster_name: the name of the raster file, could be any file that GDAL can open
        roi_name: the name of the thematic image: each pixel whose values is greater than 0 is returned
        datatype: the type of data is the thematic image (default "uint32", could be 'float32')
    Output:
        X: the sample matrix. A nXd matrix, where n is the number of referenced pixels and d is the number of variables. Each 
            line of the matrix is a pixel.
        Y: the label of the pixel
    ''' 
    
    ## Open Raster
    raster = gdal.Open(raster_name,gdal.GA_ReadOnly)
    if raster is None:
        print 'Impossible to open '+raster_name
        exit()

    ## Open ROI
    roi = gdal.Open(roi_name,gdal.GA_ReadOnly)
    if roi is None:
        print 'Impossible to open '+roi_name
        exit()

    ## Some tests
    if (raster.RasterXSize != roi.RasterXSize) or (raster.RasterYSize != roi.RasterYSize):
        print 'Images should be of the same size'
        exit()
    
    nc = roi.RasterXSize    #number of columns
    nr = roi.RasterYSize    #number of rows
    d = raster.RasterCount  #number of bands
    #GeoTransform = roi.GetGeoTransform()
    #Projection = roi.GetProjection()

    ## Get block size
    band = raster.GetRasterBand(1)
    block_sizes = band.GetBlockSize()
    x_block_size = block_sizes[0]
    y_block_size = block_sizes[1]
    del band
 
    ## Read block data
    X = sp.array([]).reshape(0,d)
    Y = sp.array([]).reshape(0,1)
 
    for i in range(0,nr,y_block_size):
        if i + y_block_size < nr: # Check for size consistency in Y
            lines = y_block_size
        else:
            lines = nr - i
        for j in range(0,nc,x_block_size): # Check for size consistency in X
            if j + x_block_size < nc:
                cols = x_block_size
            else:
                cols = nc - j
 
            # Load the reference data
            ROI = roi.GetRasterBand(1).ReadAsArray(j, i, cols, lines)
            t = sp.nonzero(ROI) #coordinates
            if t[0].size > 0:
                Y = sp.concatenate((Y,ROI[t].reshape((t[0].shape[0],1)).astype(datatype)))
                # Load the Variables
                Xtp = sp.empty((t[0].shape[0],d))
                for k in xrange(d):
                    band = raster.GetRasterBand(k+1).ReadAsArray(j, i, cols, lines)
                    Xtp[:,k] = band[t]
                try:
                    X = sp.concatenate((X,Xtp))
                except MemoryError:
                    print 'Impossible to allocate memory: ROI too big'
                    exit()


    # Clean/Close variables
    del Xtp,band    
    roi = None # Close the roi file
    raster = None # Close the raster file
    return X,Y

