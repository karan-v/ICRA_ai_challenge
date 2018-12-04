import numpy as np
import cv2
import cython
cpdef  align(unsigned short[:,:] depth, double[:,:] new_depth):
    # set the variable extension types
    cdef double  w, h, z,u1 ,v1 ,u2 ,v2
    cdef int f1,f2,j,i,g1,g2
    cdef double start,temp
    cdef double c1u ,c1v ,f1u ,f1v ,c2u ,c2v ,f2u, f2v ,x2 ,y2, z2 ,x1 ,y1 ,z1,z11,max_x,max_y
    cdef double R[3][3] ,T[3][1]
    # grab the image dimensions
    h = depth.shape[0]
    w = depth.shape[1]

    R[0][:] = [ 0.99995735, -0.0017347 , -0.00907078]
    R[1][:] = [ 0.0017159 ,  0.99999637, -0.00207944]
    R[2][:] = [ 0.00907436,  0.00206379,  0.9999567]
    T[0][0] = 0.85816367e-02
    T[1][0] = -0.0460017e-02
    T[2][0] = 0.17192013e-02
    f2u = 527.97431353
    f2v = 523.22880991
    f1u = 601.9457306
    f1v = 596.05263911
    c2u = 312.66606295
    c2v = 255.35571034
    c1u = 322.50529377
    c1v = 230.96510472 
    i=0
    u1 = 0.0
    v1 = 0.0
    max_x = 0
    max_y = 0
    # loop over the image
    while(v1<h):
        u1 = 0.0
        while(u1<w):
            f1 = int(u1)
            f2 = int(v1)
            z1 = 1.0/(-0.00307 * depth[f2][f1] + 3.33)
            x1 =  z1*(u1-c1u)/f1u
            y1 =  z1*(v1-c1v)/f1v
            x2 = R[0][0]*x1 + R[0][1]*y1 + R[0][2]*z1 + T[0][0]
            y2 = R[1][0]*x1 + R[1][1]*y1 + R[1][2]*z1 + T[1][0]
            z2 = R[2][0]*x1 + R[2][1]*y1 + R[2][2]*z1 + T[2][0]
            u2 = ((x2*f2u)/z2+c2u) 
            v2 = ((y2*f2v)/z2+c2v) 
            g1 = int(u2)
            g2 = int(v2)
            
            if(g1 < 640 and g1>0 and g2 < 480 and g2 > 0):
                
                new_depth[g2][g1] = depth[f2][f1]
                
            u1+=1
        v1+=1
    
    # return the thresholded image
    return new_depth
