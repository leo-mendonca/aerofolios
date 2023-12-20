import numpy as np
import AerofolioFino
import Malha
import ElementosFinitos
import RepresentacaoEscoamento
import time
import os
import pickle
from Definicoes import *

def teste_cilindro(n=50):
    cilindro = AerofolioFino.Cilindro(.5, 0, 1)
    nome_malha, tag_fis = Malha.malha_aerofolio(cilindro, n_pontos_contorno=n)
    Problema = ElementosFinitos.FEA(nome_malha, tag_fis)
    ux_dirichlet = [
        (Problema.nos_cont["esquerda"], lambda x: 1.),
        (Problema.nos_cont["superior"], lambda x: 0.),
        (Problema.nos_cont["inferior"], lambda x: 0.),
        (Problema.nos_cont["af"], lambda x: 0.),
    ]
    uy_dirichlet = [
        (Problema.nos_cont["esquerda"], lambda x: 0.),
        (Problema.nos_cont["superior"], lambda x: 0.),
        (Problema.nos_cont["inferior"], lambda x: 0.),
        (Problema.nos_cont["af"], lambda x: 0.),
    ]
    p_dirichlet = [(Problema.nos_cont_o1["direita"], lambda x: 0.),]
    resultados = Problema.escoamento_IPCS_Stokes(ux_dirichlet=ux_dirichlet, uy_dirichlet=uy_dirichlet, p_dirichlet=p_dirichlet, T=10, dt=0.1, Re=1, conveccao=True)
    RepresentacaoEscoamento.plotar_momento(Problema, resultados, 10)
    # RepresentacaoEscoamento.plotar_perfis(Problema, resultados, 10, lim_x=(-2,0))
    return

def numeracao_nos():
    cilindro = AerofolioFino.Cilindro(.5, 0, 1)
    nome_malha, tag_fis = Malha.malha_aerofolio(cilindro, n_pontos_contorno=10)
    Problema = ElementosFinitos.FEA(nome_malha, tag_fis)
    elem0 = Problema.elementos[0]
    x, y, z = Problema.x_nos[elem0].T
    plt.triplot(Problema.x_nos[:, 0], Problema.x_nos[:, 1], triangles=[Problema.elementos_o1[0], ], alpha=0.5)
    plt.scatter(x, y)
    for i in range(len(x)):
        plt.text(x[i], y[i], i)
    plt.show(block=False)

def teste_forca(n=20, tamanho=0.1, debug=False, executa=True):
    cilindro = AerofolioFino.Cilindro(.5, 0, 1)
    nome_malha, tag_fis = Malha.malha_aerofolio(cilindro, n_pontos_contorno=n, tamanho=tamanho)
    Problema = ElementosFinitos.FEA(nome_malha, tag_fis)
    ux_dirichlet = [
        (Problema.nos_cont["esquerda"], lambda x: 1.),
        # (Problema.nos_cont["superior"], lambda x: 0.),
        # (Problema.nos_cont["inferior"], lambda x: 0.),
        (Problema.nos_cont["af"], lambda x: 0.),
    ]
    uy_dirichlet = [
        (Problema.nos_cont["esquerda"], lambda x: 0.),
        # (Problema.nos_cont["superior"], lambda x: 0.),
        # (Problema.nos_cont["inferior"], lambda x: 0.),
        (Problema.nos_cont["af"], lambda x: 0.),
    ]
    p_dirichlet = [(Problema.nos_cont_o1["direita"], lambda x: 50.),]
    if executa:
        resultados = Problema.escoamento_IPCS_Stokes(ux_dirichlet=ux_dirichlet, uy_dirichlet=uy_dirichlet, p_dirichlet=p_dirichlet, T=10, dt=0.1, Re=1, conveccao=True)
        with open("resultados_forca.pkl", "wb") as f:
            pickle.dump((Problema, resultados), f)
    else:
        with open("resultados_forca.pkl", "rb") as f:
            Problema, resultados = pickle.load(f)
    u=resultados[10]["u"]
    p=resultados[10]["p"]
    forca, x, tensao=Problema.calcula_forcas(p,u, debug=debug)
    F=np.sum(forca,axis=0)
    x_rel=x-np.array([0.5,0])
    M=np.sum(np.cross(x_rel,forca),axis=0)
    print(f"Forca de arrasto: {F[0]}")
    print(f"Forca de sustentacao: {F[1]}")
    print(f"Momento: {M}")
    RepresentacaoEscoamento.plotar_momento(Problema, resultados, 10)
    vetor_F=forca/200
    vetor_tensao=tensao/2000
    plt.figure()
    theta=np.arange(0,2*np.pi,0.01)
    plt.plot(.5+0.5*np.cos(theta),0.5*np.sin(theta),'b-', alpha=0.3, linewidth=2)
    for i in range(len(x)):
        plt.plot([x[i,0],x[i,0]+vetor_tensao[i,0]],[x[i,1],x[i,1]+vetor_tensao[i,1]],'k-')
    plt.show(block=False)
    return forca,x

def teste_poiseuille(tamanho=0.1, p0=100, conveccao=True):
    nome_malha, tag_fis = Malha.malha_retangular("teste 5-1", tamanho, (5, 1))
    Problema = ElementosFinitos.FEA(nome_malha, tag_fis)
    ux_dirichlet = [
        (Problema.nos_cont["esquerda"], lambda x : 1.),
        (Problema.nos_cont["superior"], lambda x : 0.),
        (Problema.nos_cont["inferior"], lambda x : 0.),
    ]
    uy_dirichlet = [
        (Problema.nos_cont["esquerda"], lambda x : 0.),
        (Problema.nos_cont["superior"], lambda x : 0.),
        (Problema.nos_cont["inferior"], lambda x : 0.),
    ]
    p_dirichlet = [(Problema.nos_cont_o1["direita"], lambda x : p0),
                   # (Problema.nos_cont_o1["esquerda"], lambda x: 1.),
                   ]
    regiao_analitica = lambda x : np.logical_and(x[:, 0] >= 2, x[:, 0] < 4.9)
    solucao_analitica = lambda x : np.vstack([6 * x[:, 1] * (1 - x[:, 1]), np.zeros(len(x))]).T
    executa = True
    if executa :
        resultados = Problema.escoamento_IPCS_Stokes(ux_dirichlet=ux_dirichlet, uy_dirichlet=uy_dirichlet, p_dirichlet=p_dirichlet, T=10, dt=0.05, Re=1, solucao_analitica=solucao_analitica, regiao_analitica=regiao_analitica, conveccao=conveccao)
        with open(os.path.join("Picles", "resultados Navier-Stokes.pkl"), "wb") as f :
            pickle.dump((Problema, resultados), f)
    else :
        with open(os.path.join("Picles", "resultados.pkl"), "rb") as f :
            Problema, resultados = pickle.load(f)

    t0 = time.process_time()
    RepresentacaoEscoamento.plotar_perfis(Problema, resultados, 10)
    t1 = time.process_time()
    print(f"Perfis plotados em {t1 - t0:.4f} s")

    # plotar_momento(Problema, resultados, 3)
    RepresentacaoEscoamento.plotar_momento(Problema, resultados, 10)
    plt.show(block=False)


if __name__ == "__main__":
    teste_forca(n=40, tamanho=0.1, debug=False, executa=False)
    plt.show(block=False)
    teste_poiseuille(tamanho=0.2, p0=-100, conveccao=False)
    plt.show()