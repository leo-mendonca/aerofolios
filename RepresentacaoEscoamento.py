import numpy
from matplotlib import pyplot

from Definicoes import *


def plotar_momento(Problema, resultados, t, plotar_auxiliares=False):
    plt.figure()
    plt.suptitle(f"Velocidade horizontal - ux   t= {t} s")
    plt.triplot(Problema.x_nos[:, 0], Problema.x_nos[:, 1], Problema.elementos_o1, alpha=0.5)
    plt.scatter(Problema.x_nos[:, 0], Problema.x_nos[:, 1], c=resultados[t]["u"][:, 0])
    plt.colorbar()
    plt.figure()
    plt.suptitle(f"Velocidade vertical - uy   t= {t} s")
    plt.triplot(Problema.x_nos[:, 0], Problema.x_nos[:, 1], Problema.elementos_o1, alpha=0.5)
    plt.scatter(Problema.x_nos[:, 0], Problema.x_nos[:, 1], c=resultados[t]["u"][:, 1])
    plt.colorbar()
    if plotar_auxiliares:
        plt.figure()
        plt.suptitle(f"Velocidade horizontal - u*x   t= {t} s")
        plt.triplot(Problema.x_nos[:, 0], Problema.x_nos[:, 1], Problema.elementos_o1, alpha=0.5)
        plt.scatter(Problema.x_nos[:, 0], Problema.x_nos[:, 1], c=resultados[t]["u*"][:, 0])
        plt.colorbar()
        plt.figure()
        plt.suptitle(f"Pressao ficticia - p*   t= {t} s")
        plt.triplot(Problema.x_nos[:, 0], Problema.x_nos[:, 1], Problema.elementos_o1, alpha=0.5)
        plt.scatter(Problema.x_nos_o1[:, 0], Problema.x_nos_o1[:, 1], c=resultados[t]["p*"])
        plt.colorbar()
    plt.figure()
    plt.suptitle(f"Pressao - p   t= {t} s")
    plt.triplot(Problema.x_nos[:, 0], Problema.x_nos[:, 1], Problema.elementos_o1, alpha=0.5)
    plt.scatter(Problema.x_nos_o1[:, 0], Problema.x_nos_o1[:, 1], c=resultados[t]["p"])
    plt.colorbar()


def plotar_perfil(Problema, resultados, t, x=4, eixo=None, ordem=2):
    r = np.linspace([x, 0, 0], [x, 1, 0], 1001)
    u = np.array([Problema.interpola(p, resultados[t]["u"], ordem=ordem) for p in r])
    ux = u[:, 0]
    uy = u[:, 1]
    if eixo is None:
        plt.figure()
        plt.suptitle(f"Perfil de velocidade horizontal - ux   t= {t},  x={x}")
    plt.plot(ux, r[:, 1], label=f"ux({x},y)")


def plotar_perfis(Problema, resultados, t, lim_x=(0,5)):
    fig, eixo = plt.subplots()
    for x in np.arange(lim_x[0], lim_x[1]+.00000001, 0.5):
        plotar_perfil(Problema, resultados, t, x, eixo)
    eixo.legend()
    return

##TODO tracar linha de corrente a partir da velocidade em cada ponto

