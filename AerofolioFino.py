from Definicoes import np,plt, mcolors, mtick
from Definicoes import os


class Aerofolio(object) :
    pass


class AerofolioFino(Aerofolio) :
    '''Objeto representando um aerofolio de espessura nula e um escoamento (inviscido) incidente sobre ele'''

    def __init__(self, vetor_coeficientes, U0, alfa) :
        '''
        :param vetor_coeficientes: array de coeficientes da serie de Fourier de senos que define a linha media do aerofolio
        :param U0: velocidade do escoamento incidente
        :param alfa: angulo de ataque do aerofolio, em radianos
        '''
        self.vetor_coeficientes = vetor_coeficientes
        self.U0 = U0
        self.angulo = -alfa ##O angulo de ataque eh negativo para um escoamento de cima para baixo
        self.alfa=alfa
        A0, A1, A2 = self.calcula_coef_vorticidade(0), self.calcula_coef_vorticidade(1), self.calcula_coef_vorticidade(2)
        self.c_L = 2 * np.pi * (A0 + A1 / 2)
        self.c_M = -np.pi / 2 * (A0 + A1 - A2 / 2)
        self.c_D = 0
        self.nome = "Aerofolio-" + "-".join([str(i) for i in vetor_coeficientes])
        ###Ponto de quarto de corda em torno do qual o momento eh calculado:
        self.x_o=1/4*np.cos(self.angulo)
        self.y_o=1/4*np.sin(self.angulo)
        self.centro_aerodinamico=np.array((0.25,self.y_camber(0.25)))

    def x_med(self, eta) :
        '''Posicao horizontal da linha media do aerofolio na posicao eta (eta varia de 0 a 1 e faz as vezes de x quando o aerofolio esta horizontal)'''
        return eta * np.cos(self.angulo) - self.y_camber(eta) * np.sin(self.angulo)

    def y_med(self, eta) :
        '''Posicao vertical da linha media do aerofolio na posicao eta (eta varia de 0 a 1)'''
        return eta * np.sin(self.angulo) + self.y_camber(eta) * np.cos(self.angulo)

    def y_camber(self, x) :
        '''Camber, em funcao de x, do aerofolio'''
        return np.sum([self.vetor_coeficientes[i] * np.sin(i * np.pi * x) for i in range(len(self.vetor_coeficientes))], axis=0)

    def y_sup_0(self, x) :
        '''Linha superior do aerofolio'''
        return self.y_camber(x)  # Aerofolio sem espessura

    def y_inf_0(self, x) :
        '''Linha inferior do aerofolio'''
        return self.y_camber(x)  # Aerofolio sem espessura

    def x_sup_0(self, x) :
        '''Posicao horizontal da lina superior do aerofolio. Pode ser deslocado em caso de espessura nao-nulo'''
        return x

    def x_inf_0(self, x) :
        ''' Posicao horizontal da lina inferior do aerofolio. Pode ser deslocado em caso de espessura nao-nulo'''
        return x

    def y_sup(self, x) :
        '''Linha superior do aerofolio, considerando o angulo de inclinacao alfa'''
        return self.x_sup_0(x) * np.sin(self.angulo) + self.y_sup_0(x) * np.cos(self.angulo)

    def y_inf(self, x) :
        '''Linha inferior do aerofolio, considerando o angulo de inclinacao alfa'''
        return self.x_inf_0(x) * np.sin(self.angulo) + self.y_inf_0(x) * np.cos(self.angulo)

    def x_sup(self, x) :
        '''Posicao horizontal da lina superior do aerofolio, considerando o angulo de inclinacao alfa'''
        return self.x_sup_0(x) * np.cos(self.angulo) - self.y_sup_0(x) * np.sin(self.angulo)

    def x_inf(self, x) :
        return self.x_inf_0(x) * np.cos(self.angulo) - self.y_inf_0(x) * np.sin(self.angulo)

    def angulo_sup(self, x) :
        '''Angulo, em rad, da linha superior do aerofolio na posicao x'''
        dx = 0.0001
        dx_sup = self.x_sup(x + dx) - self.x_sup(x)
        dy_sup = self.y_sup(x + dx) - self.y_sup(x)
        angulo = np.arctan2(dy_sup, dx_sup)
        return angulo

    def angulo_inf(self, x) :
        '''Angulo, em rad, da linha inferior do aerofolio na posicao x'''
        dx = 0.0001
        dx_inf = self.x_inf(x + dx) - self.x_inf(x)
        dy_inf = self.y_inf(x + dx) - self.y_inf(x)
        angulo = np.arctan2(dy_inf, dx_inf)
        return angulo

    def calcula_coef_vorticidade(self, n) :
        '''
        Calcula o n-esimo coeficientes da serie de Fourier de vorticidade do aerofolio
        :return An
        '''
        integral = 0
        d_theta = 0.01
        range_theta = np.arange(0, np.pi, d_theta)
        if n == 0 :
            for i in range(1, len(self.vetor_coeficientes) + 1) :
                integrando = lambda theta : np.cos(i * np.pi / 2 * (1 - np.cos(theta)))
                integral += np.sum(i * self.vetor_coeficientes[i - 1] * integrando(range_theta) * d_theta)
            integral *= self.alfa
        elif n >= 1 :
            for i in range(1, len(self.vetor_coeficientes) + 1) :
                integrando = lambda theta : np.cos(i * np.pi / 2 * (1 - np.cos(theta))) * np.cos(n * theta)
                integral += np.sum(i * self.vetor_coeficientes[i - 1] * integrando(range_theta) * d_theta)
            integral *= 2
        else :
            raise ValueError("n deve ser inteiro e maior ou igual a zero")
        return integral

    def desenhar_fino(self) :
        '''Plota o aerofolio em uma janela do matplotlib como uma linha sem espessura'''
        fig, eixo = plt.subplots()
        fig.set_size_inches(7.5, 5)
        x = np.arange(0, 1.01, 0.01)
        y = self.y_camber(x)
        eixo.plot(x, y, color="black")

    def desenhar(self, eixo=None) :
        if eixo is None:
            fig, eixo = plt.subplots()
            fig.set_size_inches(7.5, 5)
            eixo.set_xlim(-0.05, 1.05)
            eixo.set_ylim(-0.2, 0.2)
            eixo.set_aspect("equal")
            eixo.set_xlabel("x")
            eixo.set_ylabel("y")
            eixo.xaxis.set_major_formatter(mtick.StrMethodFormatter("{x:.1f} D"))
            eixo.yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:.1f} D"))
        x = np.arange(0, 1.01, 0.0001)
        x1 = self.x_sup(x)
        x2 = self.x_inf(x)
        ymed = self.y_med(x)
        xmed=self.x_med(x)
        y1 = self.y_sup(x)
        y2 = self.y_inf(x)
        eixo.plot(x1, y1, color="black")
        eixo.plot(x2, y2, color="black")
        x_poli, y_poli=np.concatenate((x1, x2[::-1])), np.concatenate((y1, y2[::-1])) ##poligono fechado que repreesent o aerofolio
        eixo.fill(x_poli, y_poli, color=mcolors.CSS4_COLORS["lightgreen"], alpha=0.3)
        # eixo.fill_between(x1, y1, color=mcolors.CSS4_COLORS["lightgreen"], alpha=0.3)
        # eixo.fill_between(x2, y2, color=mcolors.CSS4_COLORS["lightgreen"], alpha=0.3)
        eixo.plot(xmed, ymed, color="gray", linestyle="dashed", alpha=0.5)


class AerofolioFinoNACA4(AerofolioFino) :
    def __init__(self, vetor_coeficientes, alfa, U0) :
        self.const_m, self.const_p, self.const_t = vetor_coeficientes
        self.theta_p = np.arccos(1 - 2 * self.const_p)
        self.beta = 1 / self.const_p ** 2 - 1 / (1 - self.const_p) ** 2
        super(AerofolioFinoNACA4, self).__init__(vetor_coeficientes, U0, alfa)
        alfa_grau=alfa*180/np.pi
        self.volume = 0.6851 * self.const_t  ##Integral de x*espessura(x) entre 0 e 1 (aproximacao!)
        self.nome = f"NACA-{(self.const_m * 100):.2f}-{(self.const_p * 10):.2f}-{(self.const_t * 100):.2f}-{int(alfa_grau)}°"

    ##TODO validar topologia problematica com autointersecao do contorno inferior quando p eh muito pequena ou t eh muito grande

    def y_camber(self, x) :
        y = (self.const_m / self.const_p ** 2 * (2 * self.const_p * x - x ** 2)) * (x < self.const_p) + (
                    self.const_m / (1 - self.const_p) ** 2 * (1 - 2 * self.const_p + 2 * self.const_p * x - x ** 2)) * (x >= self.const_p)
        return y

    def grad_y_camber(self, x) :
        grad = self.const_m / self.const_p ** 2 * (2 * self.const_p - 2 * x) * (x < self.const_p) + self.const_m / (1 - self.const_p) ** 2 * (2 * self.const_p - 2 * x) * (x >= self.const_p)
        return grad

    def theta_camber(self, x) :
        return np.arctan(self.grad_y_camber(x))

    def espessura(self, x) :
        y_espessura = 5 * self.const_t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2 + 0.2843 * x ** 3 - 0.1015 * x ** 4)
        return y_espessura

    def y_sup_0(self, x) :
        return self.y_camber(x) + self.espessura(x) * np.cos(self.theta_camber(x))

    def y_inf_0(self, x) :
        return self.y_camber(x) - self.espessura(x) * np.cos(self.theta_camber(x))

    def x_sup_0(self, x) :
        return x - self.espessura(x) * np.sin(self.theta_camber(x))

    def x_inf_0(self, x) :
        return x + self.espessura(x) * np.sin(self.theta_camber(x))

    def calcula_coef_vorticidade(self, n) :
        if n == 0 :
            integral = self.alfa- 2*self.const_m/np.pi*((self.const_p-1/2)*(np.pi+self.beta*self.theta_p) + 1/2*self.beta*np.sin(self.theta_p))
        elif n == 1 :
            integral=self.const_m/np.pi*(4*self.beta*(self.const_p-1/2)*np.sin(self.theta_p)+self.beta*(np.sin(2*self.theta_p)/2+self.theta_p)+np.pi/(1-self.const_p)**2)
        elif n > 1 :
            integral=4/(np.pi)*self.const_m*self.beta*((self.const_p-1/2)*np.sin(n*self.theta_p)/n +1/2/(1-n**2)*(np.sin(self.theta_p)*np.cos(n*self.theta_p)-n*np.sin(n*self.theta_p)*np.cos(self.theta_p)))
        else :
            raise ValueError("n deve ser inteiro e maior ou igual a zero")
        return integral

class Cilindro(AerofolioFino) :
    def __init__(self, raio, alfa, U0) :

        self.raio = raio
        self.volume = np.pi * raio ** 2
        self.U0 = U0
        self.alfa = alfa
        self.nome = f"Cilindro-{raio}-"
        self.centro_aerodinamico=np.array([0,0])

    ##Nessa classe, eta representa o angulo em torno do cilindro, e nao a coordenada x
    def x_med(self, eta):
        return 1/2-1/2*np.cos(eta*np.pi)
    def y_med(self, eta):
        return eta*0
    def x_sup_0(self, eta):
        return 1/2-1/2*np.cos(eta*np.pi)
    def x_inf_0(self, eta):
        return 1/2-1/2*np.cos(eta*np.pi)
    def y_sup_0(self, eta):
        return 1/2*np.sin(eta*np.pi)

    def y_inf_0(self, eta):
        return -1/2*np.sin(eta*np.pi)


NACA4412 = AerofolioFinoNACA4([0.04, 0.4, 0.12], 0, 1)
NACA4412_5= AerofolioFinoNACA4([0.04, 0.4, 0.12], -5*np.pi/180, 1)
NACA4412_10= AerofolioFinoNACA4([0.04, 0.4, 0.12], -10*np.pi/180, 1)


if __name__ == "__main__" :
    plt.rcParams["axes.grid"] = True
    af=AerofolioFinoNACA4([0.04,0.40,0.12], 0, 100)
    af.desenhar()
    plt.savefig(os.path.join("Saida","Aerofolio Fino NACA4","Figuras","NACA4412.png"), dpi=300, bbox_inches="tight")



    # NACA2412=AerofolioFinoNACA4([0.02,0.4,0.12], 0, 1)
    # NACA2412.desenhar()
    # NACA4412=AerofolioFinoNACA4([0.04,0.4,0.12], 0, 1)
    # NACA4412.desenhar()
    # NACA0024=AerofolioFinoNACA4([0.0,0.01,0.24], 0, 1)
    # NACA0024.desenhar()
    # NACA0050=AerofolioFinoNACA4([0.06,0.01,0.5], 0, 1)
    # plt.show(block=False)



    # # coefs=np.array([1/4,1/8,-1/16,1/32])
    # # af=AerofolioFino(coefs, 1, 0)
    # m=0.06
    # p=0.4
    # t=0.12
    # af=AerofolioFinoNACA4([m,p,t], 0, 1)
    # af.desenhar()
    # print(f"c_D = {af.c_D}")
    # print(f"c_L = {af.c_L}")
    # print(f"c_M = {af.c_M}")
    # plt.show(block=False)
    # # coefs = np.array([-1., ])
    # # af = AerofolioFino(coefs, 1, 0)
    # # print(f"c_D = {af.c_D}")
    # # print(f"c_L = {af.c_L}")
    # # print(f"c_M = {af.c_M}")
    #
    # alfas=np.arange(-45,45.1,0.1)
    # lista_c_L=[]
    # lista_c_D=[]
    # lista_c_M=[]
    # for alfa in alfas:
    #     pass
    #     af=AerofolioFinoNACA4([m,p,t], alfa, 1)
    #     lista_c_L.append(af.c_L)
    #     lista_c_D.append(af.c_D)
    #     lista_c_M.append(af.c_M)
    # plt.figure()
    # plt.plot(alfas, lista_c_L, label="c_L")
    # plt.plot(alfas, lista_c_D, label="c_D")
    # plt.plot(alfas, lista_c_M, label="c_M")
    # plt.legend()
    # plt.show(block=False)
    # plt.show(block=True)
