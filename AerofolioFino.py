import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Aerofolio(object):
    pass

class AerofolioFino(Aerofolio):
    '''Objeto representando um aerofolio de espessura nula e um escoamento (inviscido) incidente sobre ele'''
    def __init__(self, vetor_coeficientes, U0, alfa):
        '''
        :param vetor_coeficientes: array de coeficientes da serie de Fourier de senos que define a linha media do aerofolio
        :param U0: velocidade do escoamento incidente
        :param alfa: angulo de ataque do aerofolio
        '''
        self.vetor_coeficientes = vetor_coeficientes
        self.U0 = U0
        self.alfa=alfa
        A0, A1, A2 = self.calcula_coef_vorticidade(0), self.calcula_coef_vorticidade(1), self.calcula_coef_vorticidade(2)
        self.c_L=2*np.pi*(A0+A1/2)
        self.c_M=-np.pi/2*(A0+A1-A2/2)
        self.c_D=0

    def y_camber(self, x):
        '''Camber, em funcao de x, do aerofolio'''
        return np.sum([self.vetor_coeficientes[i]*np.sin(i*np.pi*x) for i in range(len(self.vetor_coeficientes))], axis=0)

    def calcula_coef_vorticidade(self, n):
        '''
        Calcula o n-esimo coeficientes da serie de Fourier de vorticidade do aerofolio
        :return An
        '''
        integral = 0
        d_theta = 0.01
        range_theta = np.arange(0, np.pi, d_theta)
        if n==0:
            for i in range(1,len(self.vetor_coeficientes)+1):
                integrando=lambda theta: np.cos(i*np.pi/2*(1-np.cos(theta)))
                integral+=np.sum(i*self.vetor_coeficientes[i-1]*integrando(range_theta)*d_theta)
            integral*=self.alfa
        elif n>=1:
            for i in range(1,len(self.vetor_coeficientes)+1):
                integrando=lambda theta: np.cos(i*np.pi/2*(1-np.cos(theta)))*np.cos(n*theta)
                integral+=np.sum(i*self.vetor_coeficientes[i-1]*integrando(range_theta)*d_theta)
            integral*=2
        else:
            raise ValueError("n deve ser inteiro e maior ou igual a zero")
        return integral

    def desenhar(self):
        '''Plota o aerofolio em uma janela do matplotlib'''
        fig,eixo=plt.subplots()
        fig.set_size_inches(7.5,5)
        x=np.arange(0,1.01,0.01)
        y=self.y_camber(x)
        eixo.plot(x,y)

class AerofolioFinoNACA4(AerofolioFino):
    def __init__(self, vetor_coeficientes, U0, alfa):
        self.const_m, self.const_p, self.const_t = vetor_coeficientes
        self.theta_p=np.arccos(1-2*self.const_p)
        self.beta=1/self.const_p**2 - 1/(1-self.const_p)**2
        super(AerofolioFinoNACA4, self).__init__(vetor_coeficientes, U0, alfa)


    def y_camber(self, x):
        y = (self.const_m/self.const_p**2*(2*self.const_p*x-x**2))*(x<self.const_p) + (self.const_m/(1-self.const_p)**2*(1-2*self.const_p+2*self.const_p*x-x**2))*(x>=self.const_p)
        return y
    def grad_y_camber(self, x):
        grad=self.const_m/self.const_p**2*(2*self.const_p-2*x)*(x<self.const_p) + self.const_m/(1-self.const_p)**2*(2*self.const_p-2*x)*(x>=self.const_p)
        return grad

    def espessura(self, x):
        y_espessura= 5*self.const_t*(0.2969*np.sqrt(x)-0.1260*x-0.3516*x**2+0.2843*x**3-0.1015*x**4)
        return y_espessura

    def calcula_coef_vorticidade(self, n):
        if n==0:
            integral=self.alfa/np.pi*self.const_m* ((2*self.const_p-1)*self.theta_p*self.beta +(2*self.const_p-1)/(1-self.const_p)**2*np.pi+ np.sin(self.theta_p)*self.beta)
        elif n==1:
            integral=self.const_m/np.pi*(self.beta*((2*self.const_p-1)*np.sin(self.theta_p)+np.sin(2*self.theta_p)/4+self.theta_p/2) +np.pi/2/(1-self.const_p)**2)
        elif n>1:
            integral=self.const_m*self.beta/np.pi*((2*self.const_p-1)/n*np.sin(n*self.theta_p)+ (np.sin(self.theta_p)*np.cos(n*self.theta_p)- n*np.cos(self.theta_p)*np.sin(n*self.theta_p))/(1-n))
        else:
            raise ValueError("n deve ser inteiro e maior ou igual a zero")
        return integral

    def desenhar_fino(self):
        super(AerofolioFinoNACA4, self).desenhar()

##TODO implementar classe NACA 5 digitos
##TODO desenhar limites do aerofolio para poder usar em Elementos Finitos




if __name__=="__main__":
    plt.rcParams["axes.grid"]=True
    # coefs=np.array([1/4,1/8,-1/16,1/32])
    # af=AerofolioFino(coefs, 1, 0)
    m=0.02
    p=0.4
    t=0.12
    af=AerofolioFinoNACA4([m,p,t], 1, 0)
    af.desenhar()
    print(f"c_D = {af.c_D}")
    print(f"c_L = {af.c_L}")
    print(f"c_M = {af.c_M}")
    plt.show(block=False)
    # coefs = np.array([-1., ])
    # af = AerofolioFino(coefs, 1, 0)
    # print(f"c_D = {af.c_D}")
    # print(f"c_L = {af.c_L}")
    # print(f"c_M = {af.c_M}")

    alfas=np.arange(-45,45.1,0.1)
    lista_c_L=[]
    lista_c_D=[]
    lista_c_M=[]
    for alfa in alfas:
        pass
        af=AerofolioFinoNACA4([m,p,t], 1, alfa)
        lista_c_L.append(af.c_L)
        lista_c_D.append(af.c_D)
        lista_c_M.append(af.c_M)
    plt.figure()
    plt.plot(alfas, lista_c_L, label="c_L")
    plt.plot(alfas, lista_c_D, label="c_D")
    plt.plot(alfas, lista_c_M, label="c_M")
    plt.legend()
    plt.show(block=False)
    plt.show(block=True)
