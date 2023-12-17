import AerofolioFino
import Malha
import ElementosFinitos
import RepresentacaoEscoamento
from Definicoes import *

def teste_cilindro():
    cilindro = AerofolioFino.Cilindro(.5, 0, 1)
    nome_malha, tag_fis = Malha.malha_aerofolio(cilindro, n_pontos_contorno=50)
    Problema = ElementosFinitos.FEA(nome_malha, tag_fis)
    ux_dirichlet = [
        (Problema.nos_cont["entrada"], lambda x: 1.),
        (Problema.nos_cont["superior"], lambda x: 0.),
        (Problema.nos_cont["inferior"], lambda x: 0.),
        (Problema.nos_cont["af"], lambda x: 0.),
    ]
    uy_dirichlet = [
        (Problema.nos_cont["entrada"], lambda x: 0.),
        (Problema.nos_cont["superior"], lambda x: 0.),
        (Problema.nos_cont["inferior"], lambda x: 0.),
        (Problema.nos_cont["af"], lambda x: 0.),
    ]
    p_dirichlet = [(Problema.nos_cont_o1["saida"], lambda x: 0.),]
    resultados = Problema.escoamento_IPCS_Stokes(ux_dirichlet=ux_dirichlet, uy_dirichlet=uy_dirichlet, p_dirichlet=p_dirichlet, T=10, dt=0.05, Re=1, conveccao=True)
    RepresentacaoEscoamento.plotar_momento(Problema, resultados, 10)
    # RepresentacaoEscoamento.plotar_perfis(Problema, resultados, 10, lim_x=(-2,0))


if __name__ == "__main__":
    teste_cilindro()
    plt.show()