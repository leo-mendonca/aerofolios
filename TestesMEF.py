import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import AerofolioFino
import Malha
import ElementosFinitos
import RepresentacaoEscoamento
import time
import os
import pickle
import zipfile
from Definicoes import *


def salvar_resultados(nome_malha, tag_fis, resultados, nome_arquivo):
    '''Salva o resultado dop caso estacionario em um arquivo comprimido .zip.
    Faz par com carregar_resultados'''
    str_tags="\n".join([f"{k}:{v}" for k,v in tag_fis.items()])
    ultimo_resultado=max(resultados.keys())
    u,p=resultados[ultimo_resultado]["u"],resultados[ultimo_resultado]["p"]
    str_u=",".join(u[:,0].astype(str))
    str_v=",".join(u[:,1].astype(str))
    str_p=",".join(p.astype(str))

    with zipfile.ZipFile(nome_arquivo, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as f:
        f.write(nome_malha, os.path.basename(nome_malha))
        f.writestr("tag_fis.csv", str_tags)
        f.writestr("u.csv", str_u)
        f.writestr("v.csv", str_v)
        f.writestr("p.csv", str_p)
        # f.writestr("Problema.pkl", pickle.dumps(Problema))
        # f.writestr("resultados.pkl", pickle.dumps(resultados))

def carregar_resultados(nome_arquivo):
    '''Carrega o resultado de um caso estacionario a partir de um arquivo comprimido .zip.
    Faz par com salvar_resultados'''
    with zipfile.ZipFile(nome_arquivo, "r") as f:
        nome_malha=f.namelist()[0]
        f.extract(nome_malha, path="Malha")
        tag_fis=f.read("tag_fis.csv").decode("utf-8")
        tag_fis=dict([linha.split(":") for linha in tag_fis.split("\n")])
        tag_fis={k:int(v) for k,v in tag_fis.items()}
        u=f.read("u.csv").decode("utf-8")
        u=np.array(u.split(","), dtype=float)
        v=f.read("v.csv").decode("utf-8")
        v=np.array(v.split(","), dtype=float)
        p=f.read("p.csv").decode("utf-8")
        p=np.array(p.split(","), dtype=float)
    nome_malha=os.path.join("Malha", nome_malha)
    u=np.stack([u,v], axis=1)
    fea=ElementosFinitos.FEA(nome_malha, tag_fis)
    return fea, u, p, nome_malha

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
    return

def teste_forca(n=20, tamanho=0.1, p0=0., debug=False, executa=True, formulacao="A"):
    cilindro = AerofolioFino.Cilindro(.5, 0, 1)
    nome_malha, tag_fis = Malha.malha_aerofolio(cilindro, n_pontos_contorno=n, tamanho=tamanho)
    Problema = ElementosFinitos.FEA(nome_malha, tag_fis)
    ux_dirichlet = [
        (Problema.nos_cont["esquerda"], lambda x: 1.),
        (Problema.nos_cont["superior"], lambda x: 1.),
        (Problema.nos_cont["inferior"], lambda x: 1.),
        (Problema.nos_cont["af"], lambda x: 0.),
    ]
    uy_dirichlet = [
        (Problema.nos_cont["esquerda"], lambda x: 0.),
        (Problema.nos_cont["superior"], lambda x: 0.),
        (Problema.nos_cont["inferior"], lambda x: 0.),
        (Problema.nos_cont["af"], lambda x: 0.),
    ]
    p_dirichlet = [(Problema.nos_cont_o1["direita"], lambda x: p0),]
    T=3
    dt=0.01
    Re=1
    if executa:
        resultados = Problema.escoamento_IPCS_Stokes(ux_dirichlet=ux_dirichlet, uy_dirichlet=uy_dirichlet, p_dirichlet=p_dirichlet, T=T, dt=dt, Re=Re, conveccao=True, u0=1., p0=p0)
        # with open("Picles/resultados_forca.pkl", "wb") as f:
        #     pickle.dump((Problema, resultados), f)
        salvar_resultados(nome_malha, tag_fis, resultados, f"Saida/Cilindro n={n} h={tamanho} dt={dt} Re={Re} {formulacao}.zip")
        RepresentacaoEscoamento.plotar_momento(Problema, resultados, T)
        u = resultados[T]["u"]
        p = resultados[T]["p"]
    else:

        # with open("Picles/resultados_forca.pkl", "rb") as f:
        #     Problema, resultados = pickle.load(f)
        Problema, u, p, nome_malha = carregar_resultados(f"Saida/Cilindro n={n} h={tamanho} dt={dt} Re={Re} {formulacao}.zip")

    forca, x, tensao=Problema.calcula_forcas(p,u, debug=debug)
    F=np.sum(forca,axis=0)
    x_rel=x-np.array([0.5,0])
    M=np.sum(np.cross(x_rel,forca),axis=0)
    print(f"Forca de arrasto: {F[0]}")
    print(f"Forca de sustentacao: {F[1]}")
    print(f"Momento: {M}")
    vetor_F=forca/200
    vetor_tensao=tensao/200
    plt.figure()
    theta=np.arange(0,2*np.pi,0.01)
    plt.plot(.5+0.5*np.cos(theta),0.5*np.sin(theta),'b-', alpha=0.3, linewidth=2)
    for i in range(len(x)):
        plt.plot([x[i,0],x[i,0]+vetor_tensao[i,0]],[x[i,1],x[i,1]+vetor_tensao[i,1]],'k-')
    (x,y),mapa_p=RepresentacaoEscoamento.mapa_de_cor(Problema,p, ordem=1, resolucao=0.05, titulo=u"Pressão")
    ##Salvando os resultados
    resolucao=0.05
    x = np.arange(Problema.x_min, Problema.x_max + resolucao, resolucao)
    y = np.arange(Problema.y_min, Problema.y_max + resolucao, resolucao)
    localizacao=Problema.localiza_grade(x,y)
    (x,y),mapa_p=RepresentacaoEscoamento.mapa_de_cor(Problema, p, ordem=1, resolucao=None, x_grade=x, y_grade=y, local_grade=localizacao, titulo=u"Pressão")
    (x, y), mapa_u = RepresentacaoEscoamento.mapa_de_cor(Problema, u[:,0], ordem=2, resolucao=None, x_grade=x, y_grade=y, local_grade=localizacao, titulo=u"Velocidade horizontal")
    (x, y), mapa_u = RepresentacaoEscoamento.mapa_de_cor(Problema, u[:, 1], ordem=2, resolucao=None, x_grade=x, y_grade=y, local_grade=localizacao, titulo=u"Velocidade vertical")
    iniciais=np.linspace([Problema.x_min, Problema.y_min+0.1], [Problema.x_min, Problema.y_max-0.1], 20)
    linhas=RepresentacaoEscoamento.linhas_de_corrente(Problema, u, pontos_iniciais=iniciais, resolucao=tamanho/10)
    plt.show(block=False)
    return forca,x

def teste_poiseuille(tamanho=0.1, p0=0, conveccao=True, Re=1., dt=0.05, T=3., executa=True, formulacao="A"):

    if executa :
        nome_malha, tag_fis = Malha.malha_retangular("teste 5-1", tamanho, (5, 1))
        Problema = ElementosFinitos.FEA(nome_malha, tag_fis)
        ux_dirichlet = [
            (Problema.nos_cont["esquerda"], lambda x: 1.),
            (Problema.nos_cont["superior"], lambda x: 0.),
            (Problema.nos_cont["inferior"], lambda x: 0.),
        ]
        uy_dirichlet = [
            (Problema.nos_cont["esquerda"], lambda x: 0.),
            (Problema.nos_cont["superior"], lambda x: 0.),
            (Problema.nos_cont["inferior"], lambda x: 0.),
        ]
        p_dirichlet = [(Problema.nos_cont_o1["direita"], lambda x: p0),
                       # (Problema.nos_cont_o1["esquerda"], lambda x: 1.),
                       ]
        regiao_analitica = lambda x: np.logical_and(x[:, 0] >= 2, x[:, 0] < 4.9)
        solucao_analitica = lambda x: np.vstack([6 * x[:, 1] * (1 - x[:, 1]), np.zeros(len(x))]).T
        resultados = Problema.escoamento_IPCS_Stokes(ux_dirichlet=ux_dirichlet, uy_dirichlet=uy_dirichlet, p_dirichlet=p_dirichlet, T=T, dt=dt, Re=Re, solucao_analitica=solucao_analitica, regiao_analitica=regiao_analitica, conveccao=conveccao, formulacao=formulacao)
        # with open(os.path.join("Picles", "resultados Poiseuille.pkl"), "wb") as f :
        #     pickle.dump((Problema, resultados), f)
        salvar_resultados(nome_malha, tag_fis, resultados, os.path.join("Saida","Poiseuille",f"Poiseuille h={tamanho} dt={dt} Re={Re} {formulacao}.zip"))
        RepresentacaoEscoamento.plotar_perfis(Problema, resultados, T)
        RepresentacaoEscoamento.plotar_momento(Problema, resultados, T)
        u=resultados[T]["u"]
        p=resultados[T]["p"]
    else :
        # with open(os.path.join("Picles", "resultados Poiseuille.pkl"), "rb") as f :
        #     Problema, resultados = pickle.load(f)
        Problema, u, p, nome_malha = carregar_resultados(os.path.join("Saida","Poiseuille",f"Poiseuille h={tamanho} dt={dt} Re={Re} {formulacao}.zip"))

    t0 = time.process_time()
    t1 = time.process_time()
    print(f"Perfis plotados em {t1 - t0:.4f} s")
    t2=time.process_time()
    resolucao=tamanho/3
    x = np.arange(Problema.x_min, Problema.x_max + resolucao, resolucao)
    y = np.arange(Problema.y_min, Problema.y_max + resolucao, resolucao)
    localizacao=Problema.localiza_grade(x,y)
    path_salvar=os.path.join("Saida","Poiseuille",f"Poiseuille h={tamanho} dt={dt} Re={Re} T={T} {formulacao}")
    (x1,y1),mapa_u=RepresentacaoEscoamento.mapa_de_cor(Problema, u[:, 0], ordem=2, resolucao=None, x_grade=x, y_grade=y, local_grade=localizacao, titulo=u"Velocidade horizontal", path_salvar=path_salvar+" u.png")
    (x1,y1),mapa_v=RepresentacaoEscoamento.mapa_de_cor(Problema, u[:, 1], ordem=2, resolucao=None, x_grade=x, y_grade=y, local_grade=localizacao, titulo=u"Velocidade vertical", path_salvar=path_salvar+" v.png")
    (x1,y1),mapa_p=RepresentacaoEscoamento.mapa_de_cor(Problema, p, ordem=1, resolucao=None, x_grade=x, y_grade=y, local_grade=localizacao, titulo=u"Pressão", path_salvar=path_salvar+" p.png")
    t3=time.process_time()
    print(f"Mapa de cor calculado em {t3-t2:.4f} s")
    t4=time.process_time()
    pontos_inicio=np.linspace([0.,0.1],[0.,0.9], 9)
    correntes=RepresentacaoEscoamento.linhas_de_corrente(Problema, u, pontos_iniciais=pontos_inicio, resolucao=resolucao)
    t5=time.process_time()
    print(f"Linhas de corrente calculadas em {t5-t4:.4f} s")


    # plotar_momento(Problema, resultados, 3)
    plt.show(block=False)

def teste_cavidade(tamanho=0.01, p0=0, conveccao=True, dt=0.01, T=3, Re=1, executa=True, formulacao="A"):

    if executa :
        nome_malha, tag_fis = Malha.malha_quadrada("cavidade", tamanho)
        Problema = ElementosFinitos.FEA(nome_malha, tag_fis)
        ux_dirichlet = [
            (Problema.nos_cont["esquerda"], lambda x: 0.),
            (Problema.nos_cont["superior"], lambda x: 1.),
            (Problema.nos_cont["inferior"], lambda x: 0.),
            (Problema.nos_cont["direita"], lambda x: 0.),
        ]
        uy_dirichlet = [
            (Problema.nos_cont["esquerda"], lambda x: 0.),
            (Problema.nos_cont["superior"], lambda x: 0.),
            (Problema.nos_cont["inferior"], lambda x: 0.),
            (Problema.nos_cont["direita"], lambda x: 0.),
        ]
        vertice_pressao = np.where(np.logical_and(Problema.x_nos[:, 0] == 1, Problema.x_nos[:, 1] == 0))[0]
        p_dirichlet = [(vertice_pressao, lambda x: p0), ]
        resultados = Problema.escoamento_IPCS_Stokes(ux_dirichlet=ux_dirichlet, uy_dirichlet=uy_dirichlet, p_dirichlet=p_dirichlet, T=T, dt=dt, Re=Re, conveccao=conveccao, formulacao=formulacao)
        salvar_resultados(nome_malha, tag_fis, resultados, os.path.join("Saida","Cavidade",f"cavidade h={tamanho} dt={dt} Re={Re} T={T} {formulacao}.zip"))
        RepresentacaoEscoamento.plotar_momento(Problema, resultados, T)
        u=resultados[T]["u"]
        p=resultados[T]["p"]
        # with open(os.path.join("Picles", f"resultados cavidade.pkl h={tamanho} dt={dt} Re={Re} {formulacao}"), "wb") as f :
        #     pickle.dump((Problema, resultados), f)
    else :
        Problema, u, p, nome_malha = carregar_resultados(os.path.join("Saida", "Cavidade",f"cavidade h={tamanho} dt={dt} Re={Re} T={T} {formulacao}.zip"))
        # with open(os.path.join("Picles", f"resultados cavidade.pkl h={tamanho} dt={dt} Re={Re} {formulacao}"), "rb") as f :
        #     Problema, resultados = pickle.load(f)
    resolucao = tamanho / 3
    x = np.arange(Problema.x_min, Problema.x_max + resolucao, resolucao)
    y = np.arange(Problema.y_min, Problema.y_max + resolucao, resolucao)
    localizacao = Problema.localiza_grade(x, y)
    path_salvar = os.path.join("Saida", "Cavidade", f"Cavidade h={tamanho} dt={dt} Re={Re} T={T} {formulacao}")
    (x1, y1), mapa_u = RepresentacaoEscoamento.mapa_de_cor(Problema, u[:, 0], ordem=2, resolucao=None, x_grade=x, y_grade=y, local_grade=localizacao, titulo=u"Velocidade horizontal", path_salvar=path_salvar + " u.png")
    (x1, y1), mapa_v = RepresentacaoEscoamento.mapa_de_cor(Problema, u[:, 1], ordem=2, resolucao=None, x_grade=x, y_grade=y, local_grade=localizacao, titulo=u"Velocidade vertical", path_salvar=path_salvar + " v.png")
    (x1, y1), mapa_p = RepresentacaoEscoamento.mapa_de_cor(Problema, p, ordem=1, resolucao=None, x_grade=x, y_grade=y, local_grade=localizacao, titulo=u"Pressão", path_salvar=path_salvar + " p.png")
    iniciais = np.linspace([0.5, 0.1], [0.5, 0.9], 10)
    ##Plotando as linhas de corrente para um lado e para o outro
    fig, eixo=plt.subplots()
    correntes = RepresentacaoEscoamento.linhas_de_corrente(Problema, u, pontos_iniciais=iniciais, resolucao=resolucao, eixo=eixo)
    correntes_inversas=RepresentacaoEscoamento.linhas_de_corrente(Problema, -u, pontos_iniciais=iniciais, resolucao=tamanho/10, eixo=eixo)
    plt.savefig(path_salvar+" correntes.png", dpi=300, bbox_inches="tight")




    plt.show(block=False)

def compara_referencia(h, dt, T, formulacao="A", plota=True):
    '''Compara os resultados de um caso estacionario com os resultados de referencia.
    Recebe como entrada um caso ja devidamente calculado'''
    arquivo_referencia="Entrada/Referencia/Cavidade solucao referencia.txt"
    valores_Re=(0.01,10,100,400,1000)
    dframe_erros=pd.DataFrame(index=valores_Re,columns=["u_med","u_rms","u_max","v_med","v_rms","v_max"], dtype=np.float64)
    roda_cores={0.01: "b", 10: "g", 100: "r", 400: "c", 1000: "m"}
    path_salvar=os.path.join("Saida","Cavidade",f"Comparacao h={h} dt={dt} T={T} {formulacao}")
    if plota:
        fig_u, eixo_u=plt.subplots()
        fig_v, eixo_v=plt.subplots()
        eixo_u.set_title(f"Velocidade horizontal em x=0.5")
        eixo_v.set_title(f"Velocidade vertical em y=0.5")
        eixo_u.set_xlabel("u")
        eixo_v.set_ylabel("v")
        eixo_u.set_ylabel("y")
        eixo_v.set_xlabel("x")
        eixo_u.set_xlim(-1,1)
        eixo_u.set_ylim(0,1)
        eixo_v.set_xlim(0,1)
        eixo_v.set_ylim(-1,1)
    for Re in valores_Re:
        arquivo_resultados=os.path.join("Saida","Cavidade",f"cavidade h={h} dt={dt} Re={Re} T={T} {formulacao}.zip")
        Problema, u, p, nome_malha = carregar_resultados(arquivo_resultados)
        dframe_ref = pd.read_csv(arquivo_referencia)
        vel_ref=dframe_ref.loc[11:,f"Re={Re}"]
        u_ref,v_ref=vel_ref.values.reshape((2,len(vel_ref)//2))
        pontos_u=np.linspace([0.5,0.0625],[0.5,0.9375], 15)
        pontos_v=np.linspace([0.0625,0.5],[0.9375,0.5], 15)
        u_calc=np.array([Problema.interpola(ponto, u, ordem=2) for ponto in pontos_u])[:,0]
        v_calc=np.array([Problema.interpola(ponto, u, ordem=2) for ponto in pontos_v])[:,1]
        erro_u=u_calc-u_ref
        u_med=np.average(erro_u)
        u_rms=np.sqrt(np.average(erro_u**2))
        u_max=np.max(np.abs(erro_u))
        erro_v=v_calc-v_ref
        v_med=np.average(erro_v)
        v_rms=np.sqrt(np.average(erro_v**2))
        v_max=np.max(np.abs(erro_v))
        dframe_erros.loc[Re]=u_med,u_rms,u_max,v_med,v_rms,v_max
        if plota:
            eixo_u.scatter(u_ref,pontos_u[:,1],marker='*', color=roda_cores[Re])
            pontos_u2 = np.linspace([0.5, 0.0625], [0.5, 0.9375], 301)
            u_calc2= np.array([Problema.interpola(ponto, u, ordem=2) for ponto in pontos_u2])[:,0]
            eixo_u.plot(u_calc2, pontos_u2[:, 1], color=roda_cores[Re], label=f"Re={Re}")
            eixo_v.scatter(pontos_v[:,0],v_ref,marker='*', color=roda_cores[Re])
            pontos_v2 = np.linspace([0.0625, 0.5], [0.9375, 0.5], 301)
            v_calc2 = np.array([Problema.interpola(ponto, u, ordem=2) for ponto in pontos_v2])[:,1]
            eixo_v.plot(pontos_v2[:,0], v_calc2,  color=roda_cores[Re], label=f"Re={Re}")
    if plota:
        eixo_u.legend()
        eixo_v.legend()
        fig_u.savefig(path_salvar+" u.png", dpi=300, bbox_inches="tight")
        fig_v.savefig(path_salvar+" v.png", dpi=300, bbox_inches="tight")
    dframe_erros.to_csv(path_salvar+" erros.csv")
    return dframe_erros


if __name__ == "__main__":
    # teste_poiseuille(tamanho=0.1, p0=0, conveccao=True, executa=True, dt=0.01, T=2, Re=1, formulacao="A")

    # teste_cavidade(tamanho=0.01, p0=0, conveccao=True, executa=False, dt=0.01, T=1, Re=1, formulacao="A")
    # for Re in (0.01,10,100,400,1000):
    #     teste_cavidade(tamanho=0.2, p0=0, conveccao=True, executa=True, dt=0.1, T=1, Re=Re, formulacao="A")
    #     plt.close("all")
    erros=compara_referencia(h=0.01, dt=0.01, T=1, formulacao="A", plota=True)
    plt.show(block=True)
    for Re in (400,1000):
        teste_cavidade(tamanho=0.01, p0=0, conveccao=True, executa=True, dt=0.01, T=1, Re=Re, formulacao="A")
        plt.close("all")
    # teste_forca(n=50, tamanho=0.3, debug=False, executa=True)
    plt.show(block=True)
    plt.close("all")
    # teste_forca(n=50, tamanho=0.3, debug=False, executa=False)
    plt.show(block=True)
    plt.show()