import matplotlib.pyplot as plt
import numpy
import numpy as np
from matplotlib import pyplot

import ElementosFinitos
from Definicoes import *


def plotar_momento(Problema, resultados, t, plotar_auxiliares=True):
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

def mapa_de_cor(Problema, variavel, ordem, resolucao=0.01, areas_excluidas=[],x_grade=None, y_grade=None, local_grade=None, plota=True, titulo="", path_salvar=None, aspecto=(6,4)):
    '''
    :param Problema:
    :param variavel: ux, uy ou p
    :param ordem: ordem da funcao de interpolacao da variavel
    :param resolucao: resolucao espacial, que se supoe ser igual para x e y
    :param areas_excluidas: lista de funcoes que retornam True para pontos que devem ser excluidos do mapa de cor
    :return:
    '''
    if not (x_grade is None or y_grade is None or local_grade is None):
        assert local_grade.shape==(len(x_grade),len(y_grade))
        x,y=x_grade,y_grade
        pontos=np.dstack(np.meshgrid(x_grade,y_grade,indexing="ij"))
        mapa = np.zeros((len(x_grade), len(y_grade)), dtype=np.float64)
        for i in range(len(x_grade)) :
            for j in range(len(y_grade)) :
                p = pontos[i,j]
                if not any([f(p) for f in areas_excluidas]):
                    mapa[i,j]=Problema.interpola_localizado(p, variavel, local_grade[i,j], ordem=ordem)
                else :
                    mapa[i, j] = np.nan
    else:
        x=np.arange(Problema.x_min,Problema.x_max+resolucao,resolucao)
        y=np.arange(Problema.y_min,Problema.y_max+resolucao,resolucao)
        mapa=np.zeros((len(x),len(y)),dtype=np.float64)
        for i in range(len(x)):
            for j in range(len(y)):
                p=np.array([x[i],y[j]])
                if not any([f(p) for f in areas_excluidas]):
                    try:
                        mapa[i,j]=Problema.interpola(p, variavel, ordem=ordem)
                    except ElementosFinitos.ElementoNaoEncontrado:
                        mapa[i,j]=np.nan
                else:
                    mapa[i,j]=np.nan
    if plota:
        fig, eixo=plt.subplots()
        plt.title(titulo)
        plot_mapa=plt.pcolormesh(x,y,mapa.T, cmap="turbo")
        # plt.triplot(Problema.x_nos[:, 0], Problema.x_nos[:, 1], Problema.elementos_o1, alpha=0.1, color="gray")
        fig.set_size_inches(aspecto)
        eixo.set_aspect("equal")
        eixocbar=fig.add_axes([eixo.get_position().x1+0.01,eixo.get_position().y0,0.02,eixo.get_position().height])
        plt.colorbar(plot_mapa,cax=eixocbar)
    if not path_salvar is None:
        plt.savefig(path_salvar, dpi=300, bbox_inches="tight")
    return (x,y),mapa


def linhas_de_corrente(Problema, u, pontos_iniciais, resolucao=0.01, areas_excluidas=[], plota=True, eixo=None):
    '''
    :param Problema:
    :param u: vetor da velocidade nos nos da malha
    :param pontos_iniciais: lista de pontos iniciais para as linhas de corrente
    :param resolucao: resolucao espacial, que se supoe ser igual para x e y
    :param areas_excluidas: lista de funcoes que retornam True para pontos que devem ser excluidos do mapa de cor
    :return:
    '''
    L=max(Problema.x_max-Problema.x_min,Problema.y_max-Problema.y_min)
    areas_excluidas= areas_excluidas+[lambda p: p[0]<Problema.x_min, lambda p: p[0]>Problema.x_max, lambda p: p[1]<Problema.y_min, lambda p: p[1]>Problema.y_max]
    linhas=[]
    for inicio in pontos_iniciais:
        linhas.append([])
        p=inicio*1
        linhas[-1].append(p*1)
        c=0
        cmax=L/resolucao*2
        while not any([f(p) for f in areas_excluidas]):
            c+=1
            try:
                vel=Problema.interpola(p, u, ordem=2)
            except ElementosFinitos.ElementoNaoEncontrado:
                break
            passo=vel/np.linalg.norm(vel)*resolucao
            p+=passo
            linhas[-1].append(p*1)
            if np.isclose(p, inicio, atol=resolucao/2).all():
                break
            if c>=cmax:
                break
        linhas[-1]=np.array(linhas[-1],dtype=np.float64)
    if plota:
        if eixo is None:
            fig,eixo=plt.subplots()
        fig=eixo.get_figure()
        eixo.set_title("Linhas de corrente")
        for linha in linhas:
            eixo.plot(linha[:,0],linha[:,1], color="black")
        eixo.set_xlim(Problema.x_min,Problema.x_max)
        eixo.set_ylim(Problema.y_min,Problema.y_max)
        eixo.set_aspect("equal")

    return linhas





