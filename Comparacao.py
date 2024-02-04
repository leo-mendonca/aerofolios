import matplotlib.pyplot as plt
import numpy as np

from Definicoes import *
import Execucao
import RedeNeural
import AerofolioFino
import TestesMEF
import keras

def compara_metodos(m,p,t,Re):
    '''Faz um grafico comparando os metodos de elementos finitos, rede neural e teoria de aerofolio fino.
    Calcula os coeficientes e um mesmo aerofolio em angulos diferentes e compara os resultados.'''
    path_saida=os.path.join("Saida","Comparacao metodos")
    alfas=np.linspace(-15,+15,1001)*np.pi/180
    alfas_mef=np.arange(-15,16,1)*np.pi/180
    vetor_entrada=np.array([[m,p,t, alfa,Re] for alfa in alfas], dtype=np.float64)
    modelo=keras.models.load_model(os.path.join("Entrada","Rede neural","Modelo.keras"))
    saidas_rede=modelo(vetor_entrada)
    saidas_af=np.zeros((len(alfas),3), dtype=np.float64)
    for i, alfa in enumerate(alfas):
        af=AerofolioFino.AerofolioFinoNACA4([m,p,t],alfa=alfa,U0=Re)
        saidas_af[i]=(af.c_L, af.c_D, af.c_M)
    saidas_mef=np.zeros((len(alfas_mef),3),dtype=np.float64)
    for i, alfa in enumerate(alfas_mef):
        af=AerofolioFino.AerofolioFinoNACA4([m,p,t],alfa=alfa,U0=Re)
        c_l,c_d,c_M=TestesMEF.teste_aerofolio(af, Re=Re, executa=False, plota_tudo=False, desenha_aerofolio=False)
        saidas_mef[i]=(c_l,c_d,c_M)
    with plt.rc_context({"text.usetex" : True}):
        fig,eixo=plt.subplots()
        eixo.plot(alfas*180/np.pi, saidas_rede[:,0], label="Rede Neural")
        eixo.plot(alfas*180/np.pi, saidas_af[:,0], label="Aerofolio Fino")
        eixo.plot(alfas_mef*180/np.pi, saidas_mef[:,0], label="Elementos Finitos")
        eixo.set_xlabel(r"$\alpha [\degree]$")
        eixo.set_ylabel(r"$C_L$")
        eixo.legend()
        plt.savefig(os.path.join(path_saida,"Comparacao_CL.png"),bbox_inches="tight",dpi=300)
        fig2,eixo2=plt.subplots()
        eixo2.plot(alfas*180/np.pi, saidas_rede[:,1], label="Rede Neural")
        eixo2.plot(alfas*180/np.pi, saidas_af[:,1], label="Aerofolio Fino")
        eixo2.plot(alfas_mef*180/np.pi, saidas_mef[:,1], label="Elementos Finitos")
        eixo2.set_xlabel(r"$\alpha [\degree]$")
        eixo2.set_ylabel(r"$C_D$")
        eixo2.legend()
        plt.savefig(os.path.join(path_saida,"Comparacao_CD.png"),bbox_inches="tight",dpi=300)
        fig3,eixo3=plt.subplots()
        eixo3.plot(alfas*180/np.pi, saidas_rede[:,2], label="Rede Neural")
        eixo3.plot(alfas*180/np.pi, saidas_af[:,2], label="Aerofolio Fino")
        eixo3.plot(alfas_mef*180/np.pi, saidas_mef[:,2], label="Elementos Finitos")
        eixo3.set_xlabel(r"$\alpha [\degree]$")
        eixo3.set_ylabel(r"$C_M$")
        eixo3.legend()
        plt.savefig(os.path.join(path_saida,"Comparacao_CM.png"),bbox_inches="tight",dpi=300)

if __name__=="__main__":
    compara_metodos(0.04,0.40,0.12,100)
