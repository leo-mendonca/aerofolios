import numpy as np

def laplaciano(f, lim_x=(0,1), lim_y=(0,1)):
    '''Calcula a derivada de uma funcao f(x,y) usando diferencas finitas.
    Recebe uma funcao f(x,y) e retorna o mapa do laplaciano de f'''
    x=np.linspace(lim_x[0], lim_x[1], num=101)
    y=np.linspace(lim_y[0], lim_y[1], num=101)
    dx=x[1]-x[0]
    dy=y[1]-y[0]
    xg,yg=np.meshgrid(x,y)
    u=f(xg,yg)
    derivx=(u[:,1:]-u[:,:-1])/dx
    derivy=(u[1:,:]-u[:-1,:])/dy
    # grad=np.dstack((derivx, derivy))
    deriv2x=(derivx[:,1:]-derivx[:,:-1])/dx
    deriv2y=(derivy[1:,:]-derivy[:-1,:])/dy
    laplaciano=deriv2x[:-2,:]+deriv2y[:,:-2]
    return laplaciano