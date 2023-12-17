# from mpi4py import MPI
# from dolfinx import log as dlog
import time
import warnings
import math
import os

import matplotlib.pyplot as plt
import numpy as np

import Malha

# dlog.set_log_level(dlog.LogLevel(0))
# from dolfinx import mesh as dmesh
# import dolfinx.io as dio
# from dolfinx import fem as dfem
# from dolfinx.fem import petsc as dpetsc
# dppetsc=dpetsc.PETSc #estamos usando o pestsc do dolfin, em vez do petsc4py (isso eh um problema?)
# ##TODO consertar petsc
# import ufl  # Unified Form Language. Linguagem para definicao de problemas de elementos finitos e forma fraca
from Definicoes import *

# MPI.COMM_WORLD permite a paralelizacao de uma mesma malha entre processadores diferentes

viscosidade = 1  # viscosidade dinamica do fluido


def exporta_valores(u, t, malha, path):
    '''Exporta os valores de u para um arquivo .csv'''
    # TODO fazer

    return


def calculo_aerofolio(aerofolio):
    '''
    Calcula as propriedades aerodinamicas de um aerofolio
    :param aerofolio: objeto da classe AerofolioFino
    '''
    ##TODO implementar

    nome_arquivo = Malha.malha_aerofolio(aerofolio, aerofolio.nome)


def coefs_aux_grad_o2(a, b, c, d, e, f, x, y):
    '''Calcula os coeficientes auxiliares "ax'", "bx'", "cx'", "ay'" "by'" e "cy'", usados para calcular o gfradiente da funcao de forma em um elemento de ordem 2.
    a,b,c,d,e,f: foat. Coeficientes da funcao de forma
    x,y: np.ndarray. Vetor de coordenadas dos 3 vertices do elemento'''
    ax = 2 * d * (x[1]) + f * (y[1])
    bx = 2 * d * (x[2]) + f * (y[2])
    cx = b
    ay = 2 * e * (y[1]) + f * (x[1])
    by = 2 * e * (y[2]) + f * (x[2])
    cy = c
    return ax, bx, cx, ay, by, cy


def coefs_aux_int_o2(a, b, c, d, e, f, x, y):
    '''Calcula os coeficientes auxiliares ap, bp, cp, dp, ep e fp, usados para calcular a integral da funcao de forma em um elemento de ordem 2.
    Recebemos os coeficientes'''
    ap = a
    bp = b * x[1] + c * y[1]
    cp = b * x[2] + c * y[2]
    dp = d * (x[1] ** 2) + e * (y[1] ** 2) + f * x[1] * y[1]
    ep = d * (x[2] ** 2) + e * (y[2] ** 2) + f * x[2] * y[2]
    fp = 2 * d * x[1] * x[2] + 2 * e * y[1] * y[2] + f * (x[1] * y[2] + x[2] * y[1])
    return ap, bp, cp, dp, ep, fp


def coefs_aux_int_o1(a, b, c, x, y):
    '''Calcula os coeficientes auxiliares ap, bp, cp, dp, ep e fp, usados para calcular a integral da funcao de forma em um elemento de ordem 1.'''
    ap = a
    bp = b * x[1] + c * y[1]
    cp = b * x[2] + c * y[2]
    return ap, bp, cp


def teste_laplace(nome_malha=None, tag_fis=None, ordem=1, n_teste=1, plota=False, gauss=False):
    '''Testa a resolucao do problema de laplace escalar'''
    print(f"Testando o caso numero {n_teste} com elementos de ordem {ordem}")
    t0 = time.process_time()
    if nome_malha is None:
        nome_malha, tag_fis = Malha.malha_quadrada("teste_laplace", 0.1, 2)
    elif tag_fis is None:
        raise ValueError("Se nome_malha for fornecido, tag_fis deve ser fornecido tambem")
    t1 = time.process_time()
    print(f"Malha gerada em {t1 - t0} segundos")
    fea = FEA(nome_malha, tag_fis)
    t2 = time.process_time()
    print(f"Objeto FEA inicializado em {t2 - t1} segundos")
    if n_teste == 1:
        funcao_exata = lambda x: x.T[0] + x.T[1] + 7
        lado_direito = lambda x: 0 * x.T[0]
    elif n_teste == 2:
        funcao_exata = lambda x: x.T[0] ** 2 - x.T[1] ** 2
        lado_direito = lambda x: 0 * x.T[0]
    elif n_teste == 3:
        ##p=x²+y² --> nabla²p=4
        ##TODO debugar por que da ainda da erro na malha linear
        funcao_exata = lambda x: (x.T[0] ** 2 + x.T[1] ** 2) / 4
        lado_direito = lambda x: x.T[0] * 0 + 1
    elif n_teste == 4:
        funcao_exata = lambda x: (1 + x.T[0] ** 2 + 2 * x.T[1] ** 2)
        lado_direito = lambda x: x.T[0] * 0 + 6
    if ordem == 1:
        nos = fea.nos_o1
    else:
        nos = fea.nos
    x_nos = fea.x_nos[nos]
    contornos_dirichlet = [(np.intersect1d(fea.nos_cont[chave], nos), funcao_exata) for chave in fea.nos_cont]  ##pega os pontos do contorno que estao na malha linear (vertices)
    pontos_dirichlet = np.unique(np.concatenate([contornos_dirichlet[i][0] for i in range(len(contornos_dirichlet))]))
    t3 = time.process_time()
    if gauss:
        procedimento1 = fea.procedimento_laplaciano_num
        procedimento2 = fea.procedimento_integracao_num
    else:
        procedimento1 = fea.procedimento_laplaciano
        procedimento2 = fea.procedimento_integracao_simples
    A_L = fea.monta_matriz(procedimento1, contornos_dirichlet, ordem=ordem)
    A_d, b_d = fea.monta_matriz_dirichlet(contornos_dirichlet, ordem=ordem)
    # if gauss:procedimento=fea.procedimento_integracao_num
    # else: procedimento=fea.procedimento_integracao_simples
    A_int = fea.monta_matriz(procedimento2, contornos_dirichlet, ordem=ordem)
    b_int = A_int @ (lado_direito(x_nos))
    A = (A_L + A_d).tocsc()
    A_array = A.toarray()
    b = b_d + b_int
    t4 = time.process_time()
    print(f"Matriz montada em {t4 - t3} segundos")
    print(f"Resolvendo sistema linear {A.shape[0]}x{A.shape[1]}")
    # p=np.linalg.solve(A,b)
    p_exato = funcao_exata(x_nos)
    ###Identificando erros construtivos na montagem das matrizes
    n_erros_interior = np.count_nonzero(~(np.isclose(A_L @ p_exato, b_int)))
    n_erros_dirichlet = np.count_nonzero(~(np.isclose(A_d @ p_exato, b_d)))
    n_erros_total = np.count_nonzero(~(np.isclose(A @ p_exato, b)))
    erros = ~np.isclose(A @ p_exato, b)
    print(f"Linhas erradas na matriz laplaciana: {n_erros_interior}\nLinhas erradas na matriz Dirichlet: {n_erros_dirichlet}\nLinhas erradas na matriz final do sistema: {n_erros_total}")

    p = ssp.linalg.spsolve(A, b)
    t5 = time.process_time()
    print(f"Sistema linear resolvido em {t5 - t4} segundos")
    erros_final = np.nonzero(~np.isclose(A @ p, b))  ##Vai ser sempre zero, a menos que o solver esteja errado
    if ordem == 1:
        x_nos = fea.x_nos[fea.nos_o1]
    else:
        x_nos = fea.x_nos

    erro = p - p_exato
    print("Erro maximo: ", np.max(erro))
    print("Erro medio: ", np.mean(erro))
    print("Erro RMS: ", np.sqrt(np.mean(erro ** 2)))
    if plota:
        fig1, eixo1 = plt.subplots()
        plt.triplot(fea.x_nos[:, 0], fea.x_nos[:, 1], fea.elementos_o1, alpha=0.3)
        plt.scatter(x_nos.T[0], x_nos.T[1], c=p, alpha=0.5)
        plt.colorbar()
        plt.savefig(os.path.join("Saida", f"teste{n_teste}_ordem{ordem}_p.png"), bbox_inches="tight")
        fig2, eixo2 = plt.subplots()
        erro_alto = np.abs(erro) > 1E-3
        plt.triplot(fea.x_nos[:, 0], fea.x_nos[:, 1], fea.elementos_o1, alpha=0.3)
        plt.scatter(x_nos.T[0], x_nos.T[1], c=erro, alpha=0.5)
        plt.colorbar()
        plt.savefig(os.path.join("Saida", f"teste{n_teste}_ordem{ordem}_erro.png"), bbox_inches="tight")
        fig3, eixo3 = plt.subplots()
        erro_baixo = np.abs(erro) <= 1E-3
        eixo3.triplot(fea.x_nos[:, 0], fea.x_nos[:, 1], fea.elementos_o1, alpha=0.3)
        # eixo3.scatter(x_nos[erro_baixo].T[0], x_nos[erro_baixo].T[1], alpha=0.5)
        plt.savefig(os.path.join("Saida", f"teste{n_teste}_ordem{ordem}_malha.png"), bbox_inches="tight")
        plt.show(block=False)

def produto_cartesiano_nodais(ux_elementos, uy_elementos, ordem=2):
    '''Calcula, em cada par j,k de nos em cada elemento da malha, o produto ux_i*uy_j dos valores nodais
    Usado para calcular o termo de conveccao do problema de Navier-Stokes
    Se ux=uy, calcula o produto cartesiano de uma mesma variavel
    :param ux_elementos: np.ndarray n_ex6. array de elementos contendo o valor de ux em cada no de cada elemento
    :return produtos: np.ndarray n_ex6x6. array de elementos contendo o produto cartesiano de ux e uy em cada par de nos de cada elemento'''
    if ordem==1: nos=3
    elif ordem==2: nos=6
    else: raise NotImplementedError("Ordem de elementos nao implementada")
    ux_grid=np.stack([ux_elementos.T for _ in range(nos)], axis=1)
    uy_grid=np.stack([uy_elementos.T for _ in range(nos)], axis=0)
    produtos=ux_grid*uy_grid
    return np.transpose(produtos, (2,0,1))

def tensor_pertencimento(elementos):
    '''Produz um tensor esparso G tal que Gilq = 1 se e somente se o no i eh o q-esimo no do elemento l'''
    posicao=[]
    valor=[]
    n_nos=int(np.max(elementos)+1)
    for l in range(elementos.shape[0]):
        for q in range(elementos.shape[1]):
            i=elementos[l,q]
            posicao.append((i,l,q))
            valor.append(1)

    tensor=tf.sparse.SparseTensor(posicao, valor, dense_shape=(n_nos, elementos.shape[0], elementos.shape[1]))
    return tensor



def calcula_termo_convectivo(produtos, tensor_convectivo, tensor_pertencimento, nos_dirichlet=[]):
    '''Calcula o vetor correspondente a Ni*uj*duk/dx para todos os valores de i
    :param u: vetor nx2 contendo os valores nodais de (ux,uy)
    :param tensor_convectivo: tensor n_ex6x6x6 cujo termo Dlijk da a contribuicao do produto dos nos j e k do elemento l quando a funcao teste eh igual a 1 no no i
    '''
    D=tensor_convectivo #Dqljk eh a componente da integral do termo convectivo no elemento l, com o q-esimo no como funcao teste, multiplicado pelo j-esimo e o k-esimo nos do mesmo elemento
    G=tensor_pertencimento #Gilq=1 se o no i eh o q-esimo no do elemento l
    # p=elementos.shape[1] #numero de nos por elemento, igual a 6 no elementos quadratico
    P=produtos #Pljk eh o produto de uj*uk no elemento l
    F=np.sum(D*P, axis=(2,3)).T #Flq eh a soma de Dqljk*Pljk sobre j e k
    integral=np.sum((G*F)._numpy(), axis=(1,2)) ##TODO checar se nao da problema fazer isso, ja que G eh um tf.SparseTensor, em vez de um objeto numpy
    integral[nos_dirichlet]*=0
    return integral






#
# class SolucaoEscoamento2:
#     def __init__(self, aerofolio, nome_malha, viscosidade=1, n_pontos_contorno=Malha.n_pontos_contorno_padrao, gerar_malha=True, caso="inviscido"):
#         self.aerofolio = aerofolio
#         self.n_pontos_contorno = n_pontos_contorno
#         self.viscosidade = viscosidade  # viscosidade cinematica do fluido
#         if gerar_malha:
#             nome_malha = Malha.malha_aerofolio(aerofolio, nome_malha, n_pontos_contorno)
#
#         self.resolve_escoamento(aerofolio, nome_malha, caso=caso)
#
#     def resolve_escoamento(self, aerofolio, nome_malha, caso="inviscido"):
#         '''Resolve o escoamento em torno de um aerofolio a partir da malha gerada pelo gmsh.
#             Retorna a funcao potencial como um campo do dolfin
#             :param aerofolio: objeto da classe AerofolioFino
#             :param malha: nome do arquivo da malha gerada pelo gmsh
#             :param caso: ("inviscido", "viscoso"). Define o tipo de escoamento
#             '''
#         y_1, y_2 = -1., 1.
#         x_1, x_2 = -2., 3.
#         self.limites = [[x_1, y_1], [x_2, y_2]]
#         U0 = aerofolio.U0
#         alfa = aerofolio.alfa
#         self.malha, self.cell_tags, self.facet_tags = dio.gmshio.read_from_msh(nome_malha, MPI.COMM_WORLD, rank=0, gdim=2)
#
#         v_cg2 = ufl.VectorElement("Lagrange", self.malha.ufl_cell(), 2, dim=2)  # elemento vetorial de Lagrange de ordem 2, ligado a velocidade
#         q_cg1 = ufl.FiniteElement("Lagrange", self.malha.ufl_cell(), 1)  # elemento escalar de Lagrange de ordem 1, ligado a pressao
#         V = dfem.FunctionSpace(self.malha, v_cg2)  # Espaco de funcao da velocidade
#         Q = dfem.FunctionSpace(self.malha, q_cg1)  # Espaco de funcao da pressao
#         self.espaco_V = V
#         self.espaco_Q = Q
#
#         if caso == "inviscido":
#
#             p_0 = 0.
#             p_1 = p_0
#             cond_p_entrada = lambda x: p_0 + x[0] * 0.  # define-se p=p_0 na entrada
#             # cond_p_saida = lambda x: p_1 + x[0] * 0.  #idem para p_1 na saida
#             # p_lateral = lambda x: p_0 + x[0]*0.   #pressao e velocidade constantesa nas laterais
#             u_0 = np.array([U0, 0.])  # velocidade de entrada vetorial
#             u_0_m = dfem.Constant(self.malha, u_0)  # velocidade de entrada como campo do dolfin
#             cond_u_entrada = lambda x: (u_0 + x[:2].T * 0.).T  # define-se u=u_0 na entrada
#             cond_u_lateral = lambda x: (u_0 + x[:2].T * 0.).T  # u nas laterais tem mesma velocidade da entrada nao perturbada, por estar afastado do obstaculo
#             cond_u_aerofolio = lambda x: (u_0 + x[:2].T * 0.).T  # Tentativa conceitualmente incorreta com velocidade nula no aerofolio
#             # Na saida, fazemos du/dx=0
#             p_in = dfem.Function(Q)
#             # p_out = dfem.Function(Q)
#             u_in = dfem.Function(V)
#             # u_out = dfem.Function(V)
#             u_lateral = dfem.Function(V)
#             u_aerofolio = dfem.Function(V)
#             p_in.interpolate(cond_p_entrada)
#             # p_out.interpolate(cond_p_saida)
#             u_in.interpolate(cond_u_entrada)
#             # u_out.interpolate(cond_u_saida)
#             u_lateral.interpolate(cond_u_lateral)
#             u_aerofolio.interpolate(cond_u_aerofolio)
#         elif caso == "viscoso":
#             p_0 = 0.
#             cond_p_entrada = lambda x: p_0 + x[0] * 0.  # define-se p=p_0 na entrada
#             u_0 = np.array([U0, 0.])  # velocidade de entrada vetorial
#             u_0_m = dfem.Constant(self.malha, u_0)  # velocidade de entrada como campo do dolfin
#             cond_u_entrada = lambda x: (u_0 + x[:2].T * 0.).T  # define-se u=u_0 na entrada
#             cond_u_lateral = lambda x: (u_0 + x[:2].T * 0.).T  # u nas laterais tem mesma velocidade da entrada nao perturbada, por estar afastado do obstaculo
#             cond_u_aerofolio = lambda x: (u_0 + x[:2].T * 0.).T  # Tentativa conceitualmente incorreta com velocidade nula no aerofolio
#             # Na saida, fazemos du/dx=0
#             p_in = dfem.Function(Q)
#             # p_out = dfem.Function(Q)
#             u_in = dfem.Function(V)
#             # u_out = dfem.Function(V)
#             u_lateral = dfem.Function(V)
#             u_aerofolio = dfem.Function(V)
#             p_in.interpolate(cond_p_entrada)
#             # p_out.interpolate(cond_p_saida)
#             u_in.interpolate(cond_u_entrada)
#             # u_out.interpolate(cond_u_saida)
#             u_lateral.interpolate(cond_u_lateral)
#             u_aerofolio.interpolate(cond_u_aerofolio)
#         else:
#             raise NotImplementedError("Apenas escoamentos inviscidos sao aceitos no momento!")
#         tdim = self.malha.topology.dim  # dimensao do espaco (no caso, 2D)
#         fdim = tdim - 1  # dimensao do contorno (no caso, 1D)
#         boundary_facets = dmesh.exterior_facet_indices(self.malha.topology)  # indices dos segmentos dos contornos
#         boundary_dofs = dfem.locate_dofs_topological(V, fdim, boundary_facets)  # indices dos graus de liberdade dos segmentos dos contornos
#         contorno_entrada = dfem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], x_1))
#         contorno_saida = dfem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], x_2))
#         contorno_superior = dfem.locate_dofs_geometrical(V, lambda x: np.isclose(x[1], y_2))
#         contorno_inferior = dfem.locate_dofs_geometrical(V, lambda x: np.isclose(x[1], y_1))
#         contornos_externos = np.concatenate([contorno_superior, contorno_inferior, contorno_entrada, contorno_saida])
#         self.contorno_aerofolio = np.setdiff1d(boundary_dofs, contornos_externos)
#         # TODO definir contorno do aerofolio geometricamente
#         ### Condicoes de contorno de dirichlet para velocidade
#         bc_entrada_V = dfem.dirichletbc(u_in, contorno_entrada)  # aplica a condicao de contorno de Dirichlet com valor u_in
#         # bc_saida_V = dfem.dirichletbc(u_out, contorno_saida)  # aplica a condicao de contorno de Dirichlet com valor u_out
#         bc_superior_V = dfem.dirichletbc(u_lateral, contorno_superior)  # aplica a condicao de contorno de Dirichlet com valor u_lateral
#         bc_inferior_V = dfem.dirichletbc(u_lateral, contorno_inferior)  # aplica a condicao de contorno de Dirichlet com valor u_lateral
#         bc_aaerofolio_V = dfem.dirichletbc(u_aerofolio, self.contorno_aerofolio)  # aplica a condicao de contorno de Dirichlet com valor u_lateral
#         condicoes_V = [bc_entrada_V, bc_superior_V, bc_inferior_V, bc_aaerofolio_V]
#         u = ufl.TrialFunction(V)
#         v = ufl.TestFunction(V)
#
#         ##Condicoes de contorno de dirichlet para a pressao
#         bc_entrada_p = dfem.dirichletbc(p_in, contorno_entrada)  # aplica a condicao de contorno de Dirichlet com valor p_in
#         condicoes_Q = [bc_entrada_p]
#         p = ufl.TrialFunction(Q)
#         q = ufl.TestFunction(Q)
#
#         n = ufl.FacetNormal(self.malha)
#
#         t_0 = 0
#         t_1 = 10
#         dt = 0.1
#         tempos = np.arange(t_0 + dt, t_1 + dt, dt)
#
#         ##No passo 0, supomos p e u constantes na extensao do escoamento
#         A_u_ast = ufl.dot(u, v) * ufl.dx
#         L_u_ast = ufl.dot(u_0_m, v) * ufl.dx
#
#         problema_inicial_u = dpetsc.LinearProblem(A_u_ast, L_u_ast, bcs=condicoes_V)
#         u_inicial = problema_inicial_u.solve()
#         p_inicial = dfem.Function(Q)
#         p_inicial.interpolate(p_in)
#         u_n = u_inicial
#         p_n = p_inicial
#         ##Variaveis de solucao
#         u_ast = dfem.Function(V)
#         p_novo = dfem.Function(Q)
#         u_novo = dfem.Function(V)
#         if caso == "inviscido":
#
#             ##Definindo matriz e vetor que representam o problema
#
#             # Passo 1
#             A_u_ast = 1 / dt * ufl.dot(u, v) * ufl.dx
#             L_u_ast = 1 / dt * ufl.dot(u_n, v) * ufl.dx - ufl.dot(ufl.dot(u_n, ufl.nabla_grad(u_n)), v) * ufl.dx - ufl.dot(ufl.grad(p_n), v) * ufl.dx
#
#             # Passo 2
#             A_p = ufl.dot(ufl.grad(p), ufl.grad(q)) * ufl.dx - q * ufl.dot(ufl.grad(p), n) * ufl.ds
#             L_p = ufl.dot(ufl.grad(p_n), ufl.grad(q)) * ufl.dx - q * ufl.dot(ufl.grad(p_n), n) * ufl.ds - 1 / dt * q * ufl.div(u_ast) * ufl.dx
#
#             # Passo 3
#             A_u = 1 / dt * ufl.dot(u, v) * ufl.dx
#             L_u = 1 / dt * ufl.dot(u_ast, v) * ufl.dx + ufl.dot(ufl.grad(p_n), v) * ufl.dx - ufl.dot(ufl.grad(p_novo), v) * ufl.dx
#
#         elif caso == "viscoso":
#             def epsilon(x):
#                 return ufl.sym(ufl.nabla_grad(x))
#
#             def sigma(x, y):
#                 return 2 * self.viscosidade * epsilon(x) - y * ufl.Identity(2)
#
#             u_intermediario = 0.5 * (u_n + u)  # velocidade intermediaria entre u_n anterior e u_ast
#
#             F = 1 / dt * ufl.dot(u, v) * ufl.dx - 1 / dt * ufl.dot(u_n, v) * ufl.dx + ufl.dot(ufl.dot(u_n, ufl.nabla_grad(u_n)), v) * ufl.dx + ufl.inner(sigma(u_intermediario, p_n), epsilon(v)) * ufl.dx + ufl.dot(p_n * n, v) * ufl.ds - ufl.dot(self.viscosidade * ufl.nabla_grad(u_n) * n, v) * ufl.ds
#             A_u_ast = ufl.lhs(F)
#             L_u_ast = ufl.rhs(F)
#
#             # Passo 2
#             A_p = ufl.dot(ufl.grad(p), ufl.grad(q)) * ufl.dx - q * ufl.dot(ufl.grad(p), n) * ufl.ds
#             L_p = ufl.dot(ufl.grad(p_n), ufl.grad(q)) * ufl.dx - q * ufl.dot(ufl.grad(p_n), n) * ufl.ds - 1 / dt * q * ufl.div(u_ast) * ufl.dx
#             # Passo 3
#             A_u = 1 / dt * ufl.dot(u, v) * ufl.dx
#             L_u = 1 / dt * ufl.dot(u_ast, v) * ufl.dx + ufl.dot(ufl.grad(p_n), v) * ufl.dx - ufl.dot(ufl.grad(p_novo), v) * ufl.dx
#
#         else:
#             raise NotImplementedError("Apenas escoamentos inviscidos sao aceitos no momento!")
#         ##Passo 1
#         bilinear1 = dfem.form(A_u_ast)
#         linear1 = dfem.form(L_u_ast)
#         A1 = dpetsc.assemble_matrix(bilinear1, bcs=condicoes_V)
#         A1.assemble()
#         b1 = dpetsc.create_vector(linear1)
#         solver1 = dppetsc.KSP().create(self.malha.comm)
#         solver1.setOperators(A1)
#         solver1.setType(dppetsc.KSP.Type.PREONLY)
#         solver1.getPC().setType(dppetsc.PC.Type.LU)
#         ##Passo 2
#         bilinear2 = dfem.form(A_p)
#         A2 = dpetsc.assemble_matrix(bilinear2, bcs=condicoes_Q)
#         A2.assemble()
#         linear2 = dfem.form(L_p)
#         b2 = dpetsc.create_vector(linear2)
#         solver2 = dppetsc.KSP().create(self.malha.comm)
#         solver2.setOperators(A2)
#         solver2.setType(dppetsc.KSP.Type.PREONLY)
#         solver2.getPC().setType(dppetsc.PC.Type.LU)
#         ##Passo 3
#         bilinear3 = dfem.form(A_u)
#         A3 = dpetsc.assemble_matrix(bilinear3, bcs=condicoes_V)
#         A3.assemble()
#         linear3 = dfem.form(L_u)
#         b3 = dpetsc.create_vector(linear3)
#         solver3 = dppetsc.KSP().create(self.malha.comm)
#         solver3.setOperators(A3)
#         solver3.setType(dppetsc.KSP.Type.PREONLY)
#         solver3.getPC().setType(dppetsc.PC.Type.LU)
#         # Escrevendo os resultados em arquivo
#         vtk_u = dio.VTKFile(self.malha.comm, "Saida/EF1/u.vtk", "w")
#         vtk_p = dio.VTKFile(self.malha.comm, "Saida/EF1/p.vtk", "w")
#         vtk_u.write_mesh(self.malha)
#         vtk_u.write_function(u_n, 0)
#         vtk_p.write_mesh(self.malha)
#         vtk_p.write_function(p_n, 0)
#         for t in tempos:
#             print(t)
#             ##Passo 1: encontrar a velocidade tentativa u_ast (ast=asterisco)
#             ##Tentativa de reutilizar as matrizes de solucao
#             # Update the right hand side reusing the initial vector
#             with b1.localForm() as loc_b:
#                 loc_b.set(0)
#             dpetsc.assemble_vector(b1, linear1)
#             # Apply Dirichlet boundary condition to the vector
#             dpetsc.apply_lifting(b1, [bilinear1], [condicoes_V])
#             b1.ghostUpdate(addv=dppetsc.InsertMode.ADD_VALUES, mode=dppetsc.ScatterMode.REVERSE)
#             dfem.set_bc(b1, condicoes_V)
#             # Solve linear problem
#             solver1.solve(b1, u_ast.vector)
#             u_ast.x.scatter_forward()
#             ##Passo 2: encontrar a pressao no passo n+1
#             with b2.localForm() as loc_b:
#                 loc_b.set(0)
#             dpetsc.assemble_vector(b2, linear2)
#             dpetsc.apply_lifting(b2, [bilinear2], [condicoes_Q])
#             b2.ghostUpdate(addv=dppetsc.InsertMode.ADD_VALUES, mode=dppetsc.ScatterMode.REVERSE)
#             dfem.set_bc(b2, condicoes_Q)
#             solver2.solve(b2, p_novo.vector)
#             p_novo.x.scatter_forward()
#
#             ##Passo 3: encontrar a velocidade no passo n+1
#             with b3.localForm() as loc_b:
#                 loc_b.set(0)
#             dpetsc.assemble_vector(b3, linear3)
#             dpetsc.apply_lifting(b3, [bilinear3], [condicoes_V])
#             b3.ghostUpdate(addv=dppetsc.InsertMode.ADD_VALUES, mode=dppetsc.ScatterMode.REVERSE)
#             dfem.set_bc(b3, condicoes_V)
#             solver3.solve(b3, u_novo.vector)
#             u_novo.x.scatter_forward()
#
#             u_n.interpolate(u_novo)
#             p_n.interpolate(p_novo)
#             ##Escrevendo os resultados em arquivo
#             if np.isclose(t % 0.5, 0):
#                 vtk_u.write_function(u_n, t)
#                 vtk_p.write_function(p_n, t)
#
#         # ##Definicao do solver nao linear por metodo de Newton
#         # solver = dnls.NewtonSolver(MPI.COMM_WORLD, problema)
#         # solver.convergence_criterion = "incremental"
#         # solver.rtol = 1e-6
#         # solver.report=True
#         # n_it, convergencia = problema.solve()  # resolve o sistema pelo metodo nao-linear
#         ##Armazenando os resultados como atributos da classe
#         self.u = u_n
#         self.p = p_n
#         self.val_u = self.u.vector.array
#         self.val_p = self.p.vector.array
#         self.x = self.malha.geometry.x
#
#     def interpola_solucao(self, n_pontos):
#         '''Calcula a solucao da funcao potencial em uma malha regular de pontos'''
#         # TODO implementar sem usar funcao potencial
#         raise NotImplementedError("Funcao ainda esta configurada para usar escoamento potencial")
#         x_1, y_1 = self.limites[0]
#         x_2, y_2 = self.limites[1]
#         self.malha_solucao = dmesh.create_rectangle(MPI.COMM_WORLD, [[x_1, y_1], [x_2, y_2]], [n_pontos, n_pontos], dmesh.CellType.quadrilateral)
#         espaco_solucao = dfem.FunctionSpace(self.malha_solucao, ("CG", 1))
#         self.phi_solucao = dfem.Function(espaco_solucao)
#         self.phi_solucao.interpolate(self.phi)
#         self.x_solucao = self.malha_solucao.geometry.x
#         return self.x_solucao, self.phi_solucao.vector.array
#
#     def ordena_contorno(self):
#         '''Ordena os pontos do contorno do aerofolio em sentido anti-horario'''
#         x_aerofolio = self.x[self.contorno_aerofolio]
#         x, y = x_aerofolio[:, 0], x_aerofolio[:, 1]
#
#         eta = np.arange(1, self.n_pontos_contorno) / self.n_pontos_contorno
#         x_0, y_0 = self.aerofolio.x_med(0), self.aerofolio.y_med(0)  # inicial
#         x_f, y_f = self.aerofolio.x_med(1), self.aerofolio.y_med(1)  # final
#         x_sup, y_sup = self.aerofolio.x_sup(eta), self.aerofolio.y_sup(eta)
#         x_inf, y_inf = self.aerofolio.x_inf(eta), self.aerofolio.y_inf(eta)
#         pos_inf = np.array([x_inf, y_inf]).T
#         pos_sup = np.array([x_sup, y_sup]).T
#         caminho = np.concatenate([((x_0, y_0),), pos_inf, ((x_f, y_f),), pos_sup[::-1]])  # comeca no ponto 0, faz o caminho por baixo e depois volta pór cima (sentido anti-horario)
#         pontos = np.zeros(len(caminho), dtype=int)
#         for i in range(len(pontos)):
#             pontos[i] = np.argmin((x - caminho[i, 0]) ** 2 + (y - caminho[i, 1]) ** 2)  # indice de cada ponto do contorno na lista dos pontos de contorno
#         self.indices_contorno = self.contorno_aerofolio[pontos]  # indice de cada ponto na lista global de pontos da malha
#         return self.indices_contorno
#
#     def calcula_forcas(self):
#         '''Calcula as forcas de sustentacao e arrasto e momento produzidas pela pressao'''
#         # TODO implementar sem usar funcao potencial
#         raise NotImplementedError("Funcao ainda esta configurada para usar escoamento potencial")
#         lista_pontos = np.concatenate((self.indices_contorno, self.indices_contorno[0:1]))  # da um loop completo, repetindo o ponto zero
#         x, y, z = self.x[lista_pontos].T
#         dx = x[1:] - x[:-1]
#         dy = y[1:] - y[:-1]  # o vetor entre um ponto e o seguinte eh (dx,dy)
#         ds = np.sqrt(dx ** 2 + dy ** 2)  # comprimento de cada segmento
#         dphi = self.val_phi[lista_pontos[1:]] - self.val_phi[lista_pontos[:-1]]
#         ##pontos medios de cada segmento:
#         x_med = (x[1:] + x[:-1]) / 2
#         y_med = (y[1:] + y[:-1]) / 2
#         u = dphi / ds  # modulo da velocidade em cada ponto
#         # vetor normal ao segmento entre pontos consecutivos
#         normal = np.transpose([-dy / ds, dx / ds])
#         ##Bernoulli: p/rho+1/2*U²=cte
#         ##Todas as grandezas aqui retornadas sao divididas por unidade de massa (F_L/rho, F_D/rho, M/rho)
#         ##Defina p/rho=0 no ponto 0
#         pressao_total = 0 + 1 / 2 * u[0] ** 2
#         pressao = pressao_total - 1 / 2 * u ** 2
#         forcas = (pressao * ds * normal.T).T
#         x_rel, y_rel = x_med - self.aerofolio.x_o, y_med - self.aerofolio.y_o  # posicao do ponto central dos segmentos em relacao ao centro (quarto de corda) do aerofolio
#         momentos = x_rel * forcas[:, 1] - y_rel * forcas[:, 0]  # momento em relacao ao centro do aerofolio
#         self.forca = forcas.sum(axis=0)
#         self.F_D, self.F_L = self.forca  # forcas de sustentacao e arrasto por unidade de massa especifica do fluido
#         self.M = momentos.sum()  # momento em relacao ao centro do aerofolio por unidade de massa especifica do fluido
#         return self.F_L, self.F_D, self.M
#
#     def linha_corrente(self, ponto_inicial):
#         ##TODO tracar linha de corrente a partir da velocidade em cada ponto
#         pass


class FEA(object):
    '''Classe para resolucao do escoamento usando o metodo de elementos finitos de Galerkin manualmente'''

    def __init__(self, nome_malha, tag_fis, aerofolio=None, velocidade=None):
        '''
        :param nome_malha: Nome do arquivo .msh produzido pelo gmsh. Deve ser uma malha de ordem 2
        :param tag_fis: dicionario contendo a tag de cada grupo fisico da malha
        :param aerofolio: AerofolioFino dado como entrada na simulacao
        :param velocidade: velocidade do escoamento, caso seja diferente daquela prevista pelo aerofolio
        '''
        if velocidade is None:
            if aerofolio is None:
                velocidade = 1
            else:
                velocidade = aerofolio.U0
        self.velocidade = velocidade
        self.aerofolio = aerofolio
        self.nos, self.x_nos, self.elementos, self.nos_cont, self.x_cont = Malha.ler_malha(nome_malha, tag_fis)
        self.nos_o1, self.elementos_o1 = Malha.reduz_ordem(self.elementos)
        self.x_nos_o1 = self.x_nos[self.nos_o1]
        self.nos_cont_o1 = {chave: np.intersect1d(self.nos_o1, self.nos_cont[chave]) for chave in self.nos_cont.keys()}
        self.nos_faces = np.setdiff1d(self.nos, self.nos_o1)
        self.coefs_a_lin, self.coefs_b_lin, self.coefs_c_lin, self.dets_lin = self.funcoes_forma(ordem=1)  # coeficientes das funcoes de forma lineares
        self.coefs_o1 = np.dstack([self.coefs_a_lin, self.coefs_b_lin, self.coefs_c_lin])
        self.coefs_o2 = self.funcoes_forma(ordem=2)
        self.coefs_o1_alfa = self.funcoes_forma_alfa(ordem=1)
        self.coefs_o2_alfa = self.funcoes_forma_alfa(ordem=2)
        ##Tensor de pertencimento: indica a que elementos pertence cada no, e em que posicao
        self.pertencimento=tensor_pertencimento(self.elementos)
        self.pertencimento_o1 = tensor_pertencimento(self.elementos_o1)
        ##Fatores que auxiliam na integracao de qualquer expressao que possa ser escrita como rho1+ rho2*alfa + rho3*beta + rho4*alfa² ...
        self.fatores_integracao = np.array([[math.factorial(k) * math.factorial(n) / math.factorial(k + n + 2) for k in range(7)] for n in range(7)])
        self.fatores_rho = np.array([
            self.fatores_integracao[0, 0],
            self.fatores_integracao[1, 0],
            self.fatores_integracao[0, 1],
            self.fatores_integracao[2, 0],
            self.fatores_integracao[0, 2],
            self.fatores_integracao[1, 1],
            self.fatores_integracao[3, 0],
            self.fatores_integracao[0, 3],
            self.fatores_integracao[2, 1],
            self.fatores_integracao[1, 2],
            self.fatores_integracao[4, 0],
            self.fatores_integracao[0, 4],
            self.fatores_integracao[3, 1],
            self.fatores_integracao[2,2],
            self.fatores_integracao[1, 3],
            self.fatores_integracao[5, 0],
            self.fatores_integracao[0, 5],
            self.fatores_integracao[4, 1],
            self.fatores_integracao[3, 2],
            self.fatores_integracao[2, 3],
            self.fatores_integracao[1, 4],

        ])

    def matriz_laplaciano_escalar2(self, contornos_dirichlet=[], contornos_neumann=[], contornos_livres=[], ordem=1, verbose=False, gauss=False):
        '''Monta a matriz A do sistema linear correspondente a uma equacao de Laplace nabla^2(p)=0, com p escalar
        :param contornos_dirichlet: lista de tuplas contendo, cada uma: (vetor de nos com condicao de contorno de Dirichlet, funcao u=f(x) que define a condicao de contorno)
        :param contornos_neumann: lista de tuplas contendo, cada uma: (vetor de nos com condicao de contorno de Neumann, constante u*n=k que define a condicao de contorno)
        :param contornos_livres: lista de vetores contendo os nos sem condicao de contorno definida
        :param gauss: bool. Se verdadeiro, a integracao eh feita por quadratura gaussiana, em vez do calculo analitico
        '''
        area = self.dets_lin / 2  # area de cada elemento
        if ordem == 1:
            elementos = self.elementos_o1
            nos = self.nos_o1
        elif ordem == 2:
            elementos = self.elementos
            nos = self.nos
        else:
            raise ValueError(f"Elementos de ordem {ordem} nao sao suportados")
        if verbose: print("Montando sistema linear")
        ##Teste: usar matriz esparsa coo_array para construir a matriz que define o sistema linear
        linhas, colunas, valores = [], [], []
        # A = np.zeros((len(nos), len(nos)), dtype=np.float64)
        b = np.zeros(len(nos), dtype=np.float64)
        pontos_dirichlet = np.concatenate([contornos_dirichlet[i][0] for i in range(len(contornos_dirichlet))])  # lista de todos os pontos com condicao de dirichlet
        pontos_sem_dirichlet = np.setdiff1d(nos, pontos_dirichlet)  # lista de todos os pontos sem condicao de dirichlet
        ##TODO paralelizar loop principal abaixo
        for i in pontos_sem_dirichlet:  # aqui, varremos todos os pontos nos quais a funcao teste nao eh identicamente nula
            elementos_i = np.where(elementos == i)[0]  # lista de elementos que contem o ponto i
            for l in elementos_i:  # varremos cada elemento no qual N_i != 0
                pos_i = np.nonzero(elementos[l] == i)[0][0]  # posicao do ponto i no elemento l
                ind_i = np.nonzero(nos == i)[0][0]  # indice do ponto i no vetor de nos
                if ordem == 2:
                    x_abs, y_abs, z_abs = self.x_nos[elementos[l]].T
                    x, y = x_abs - x_abs[0], y_abs - y_abs[0]
                    ai, bi, ci, di, ei, fi = self.coefs_o2[l, pos_i]
                    axi, bxi, cxi, ayi, byi, cyi = coefs_aux_grad_o2(ai, bi, ci, di, ei, fi, x, y)

                for j in elementos[l]:  # varremos os pontos do elemento l
                    pos_j = np.nonzero(elementos[l] == j)[0][0]  # posicao do ponto j no elemento l
                    ind_j = np.nonzero(nos == j)[0][0]  # indice do ponto j no vetor de nos
                    if gauss:
                        expressao = lambda x: np.dot(self.grad_N(i, l, x, ordem), self.grad_N(j, l, x, ordem))
                        valor = self.integrar_expressao(expressao, l, ordem)

                    elif ordem == 1:
                        valor = area[l] * (self.coefs_b_lin[l, pos_i] * self.coefs_b_lin[l, pos_j] + self.coefs_c_lin[l, pos_i] * self.coefs_c_lin[l, pos_j])
                        # A[ind_i,ind_j]+=valor
                    elif ordem == 2:
                        # Nesse caso, ind_i=i, porque a malha completa ja eh de ordem 2 (a menos que isso mude)
                        aj, bj, cj, dj, ej, fj = self.coefs_o2[l, pos_j]
                        ##Aqui, para calcular a integral, fazemos a mudanca de variaveis (x,y)-->(alfa,beta), onde x=(1-alfa-beta)*x1+alfa*x2+beta*x3, y=(1-alfa-beta)*y1+alfa*y2+beta*y3
                        axj, bxj, cxj, ayj, byj, cyj = coefs_aux_grad_o2(aj, bj, cj, dj, ej, fj, x, y)
                        ##Definimos um vetor rho tal que grad(Ni)*grad(Nj)= rho1 + rho2*alfa + rho3*beta + rho4*alfa² + rho5*beta² + rho6*alfa*beta
                        rho = [cxi * cxj + cyi * cyj,
                               axi * cxj + axj * cxi + ayi * cyj + ayj * cyi,
                               bxi * cxj + bxj * cxi + byi * cyj + byj * cyi,
                               axi * axj + ayi * ayj,
                               bxi * bxj + byi * byj,
                               axi * bxj + axj * bxi + ayi * byj + ayj * byi]
                        ##O fator que multiplica cada entrada de rho eh dado por J_kn = k!n!/(k+n+2)!, onde k e n sao as potencias de alfa e beta, respectivamente
                        valor = 2 * area[l] * (rho[0] / 2 + rho[1] / 6 + rho[2] / 6 + rho[3] / 12 + rho[4] / 12 + rho[5] / 24)  ##TODO conferir fator 2
                    linhas.append(ind_i)
                    colunas.append(ind_j)
                    valores.append(valor)
        ###condicoes de dirichlet:
        for (cont, funcao) in contornos_dirichlet:
            for i in cont:
                ind_i = np.nonzero(nos == i)[0][0]  # indice do ponto i no vetor de nos (se ordem=1, o vetor de nos usado eh menor que o vetor total, se a malha tiver ordem 2)
                # A[ind_i, ind_i] = 1
                linhas.append(ind_i)
                colunas.append(ind_i)
                valores.append(1)
                b[ind_i] = funcao(self.x_nos[i])
        ###condicoes de neumann: #TODO implementar caso grad!=0
        assert len([item for item in contornos_neumann if item[1] != 0]) == 0, "Condicoes de Neumann nao implementadas"  # se alguma condicao de von Neumann for nao homogenea, damos raise
        ###condicoes livres: #TODO implementar caso geral
        assert len(contornos_livres) == 0, "Condicoes livres nao implementadas"  # se houver algum ponto sem condicao de contorno, damos raise

        ##Geramos a matriz esparsa com base nos arrays de linhas, colunas e valores
        A_esparso = ssp.coo_matrix((valores, (linhas, colunas)), shape=(len(nos), len(nos)), dtype=np.float64)
        # A_esp_bsr=A_esparso.tobsr() #convetendo para o tipo BSR, que pode ser usado em operacoes de algebra linear
        # A = A_esparso.toarray()
        A = A_esparso.tobsr()
        ##Verificando se A eh singular
        # assert np.linalg.matrix_rank(A) == len(nos), "Matriz A singular" ##Verificar se a matriz tem posto cheio gasta muito tempo de processamento
        if verbose: print("Sistema linear montado")
        return A, b

    def procedimento_laplaciano(self, l, pos_i, pos_j, x, y, ordem_i, ordem_j):
        '''Procedimento elementar para gerar cada entrada da matriz da forma variacional correspondente ao laplaciano
        :param l: elemento em que estamos
        :param pos_i: posicao do ponto i no elemento l
        :param pos_j: posicao do ponto j no elemento l
        :param ordem_i: ordem do no i (1 ou 2). Corresponde tipicamente a ordem da funcao TESTE
        :param ordem_j: ordem do no j (1 ou 2). Corresponde tipicamente a ordem da funcao TENTATIVA
        '''
        area = self.dets_lin[l] / 2
        if ordem_i == 1 and ordem_j == 1:
            valor = -2 * area * (self.coefs_b_lin[l, pos_i] * self.coefs_b_lin[l, pos_j] + self.coefs_c_lin[l, pos_i] * self.coefs_c_lin[l, pos_j])
        elif ordem_i == 2 and ordem_j == 2:
            ai, bi, ci, di, ei, fi = self.coefs_o2[l, pos_i]
            axi, bxi, cxi, ayi, byi, cyi = coefs_aux_grad_o2(ai, bi, ci, di, ei, fi, x, y)
            aj, bj, cj, dj, ej, fj = self.coefs_o2[l, pos_j]
            ##Aqui, para calcular a integral, fazemos a mudanca de variaveis (x,y)-->(alfa,beta), onde x=(1-alfa-beta)*x1+alfa*x2+beta*x3, y=(1-alfa-beta)*y1+alfa*y2+beta*y3
            axj, bxj, cxj, ayj, byj, cyj = coefs_aux_grad_o2(aj, bj, cj, dj, ej, fj, x, y)
            ##Definimos um vetor rho tal que grad(Ni)*grad(Nj)= rho1 + rho2*alfa + rho3*beta + rho4*alfa² + rho5*beta² + rho6*alfa*beta
            rho = np.array([cxi * cxj + cyi * cyj,
                            axi * cxj + axj * cxi + ayi * cyj + ayj * cyi,
                            bxi * cxj + bxj * cxi + byi * cyj + byj * cyi,
                            axi * axj + ayi * ayj,
                            bxi * bxj + byi * byj,
                            axi * bxj + axj * bxi + ayi * byj + ayj * byi])
            ##O fator que multiplica cada entrada de rho eh dado por J_kn = k!n!/(k+n+2)!, onde k e n sao as potencias de alfa e beta, respectivamente
            valor = -2 * area * (rho * self.fatores_rho[:len(rho)]).sum(axis=0)
        else:
            raise NotImplementedError(f"Nao implementado calculo do laplaciano com teste ordem {ordem_i} e tentativa ordem {ordem_j}")
        return valor

    def procedimento_integracao_simples(self, l, pos_i, pos_j, x, y, ordem_i, ordem_j):
        '''Procedimento elementar que quando aplicado a um vetor de valores nodais de f(x), retorna a integral de f(x)*v'''
        det = self.dets_lin[l]
        # if x[0]>0 or y[0]>0:
        #     raise ValueError("Coordenadas do elemento nao estao modificadas para x0 ser 0")
        if ordem_i == 1 and ordem_j == 1:
            ai, bi, ci = self.coefs_o1[l, pos_i]
            aj, bj, cj = self.coefs_o1[l, pos_j]
            ai, bi, ci = coefs_aux_int_o1(ai, bi, ci, x, y)
            aj, bj, cj = coefs_aux_int_o1(aj, bj, cj, x, y)
            rho = np.array([
                ai * aj,
                ai * bj + aj * bi,
                ai * cj + aj * ci,
                bi * bj,
                ci * cj,
                bi * cj + bj * ci,
            ])

        elif ordem_i == 2 and ordem_j == 2:
            a0i, b0i, c0i, d0i, e0i, f0i = self.coefs_o2[l, pos_i]
            a0j, b0j, c0j, d0j, e0j, f0j = self.coefs_o2[l, pos_j]
            ai, bi, ci, di, ei, fi = coefs_aux_int_o2(a0i, b0i, c0i, d0i, e0i, f0i, x, y)
            aj, bj, cj, dj, ej, fj = coefs_aux_int_o2(a0j, b0j, c0j, d0j, e0j, f0j, x, y)
            rho = np.array([
                ai * aj,
                ai * bj + aj * bi,
                ai * cj + aj * ci,
                ai * dj + aj * di + bi * bj,
                ai * ej + aj * ei + ci * cj,
                ai * fj + aj * fi + bi * cj + bj * ci,
                bi * dj + bj * di,
                ci * ej + cj * ei,
                ci * dj + cj * di + bi * fj + bj * fi,
                bi * ej + bj * ei + ci * fj + cj * fi,
                di * dj,
                ei * ej,
                di * fj + dj * fi,
                di * ej + dj * ei + fi * fj,
                ei * fj + ej * fi,
            ])
        else:
            raise NotImplementedError(f"Nao implementado calculo da integral com teste ordem {ordem_i} e tentativa ordem {ordem_j}")
        valor = det * (rho * self.fatores_rho[:len(rho)]).sum(axis=0)
        return valor

    def procedimento_derivx(self, l, pos_i, pos_j, x, y, ordem_i, ordem_j):
        '''Procedimento elementar que quando aplicado a um vetor de valores nodais de f(x), retorna a integral de dNj(x)/dx*vi'''
        det = self.dets_lin[l]
        if ordem_i == 1 and ordem_j == 2:
            ##Esse eh o caso relevante para a equacao da divergencia
            api, bpi, cpi = self.coefs_o1_alfa[pos_i]
            aj, bj, cj, dj, ej, fj = self.coefs_o2[l, pos_j]
            rho = np.array([
                api * bj,
                api * (2 * dj * x[1] + fj * y[1]) + bpi * bj,
                api * (2 * dj * x[2] + fj * y[2]) + cpi * bj,
                bpi * (2 * dj * x[1] + fj * y[1]),
                cpi * (2 * dj * x[2] + fj * y[2]),
                bpi * (2 * dj * x[2] + fj * y[2]) + cpi * (2 * dj * x[1] + fj * y[1]),
            ])
            # Integracao de rho0+rho1*alfa+rho2*beta+rho3*alfa²+rho4*beta²+rho5*alfa*beta
            valor = det * (rho * self.fatores_rho[:len(rho)]).sum(axis=0)
        elif ordem_i == 2 and ordem_j == 1:
            api, bpi, cpi, dpi, epi, fpi = self.coefs_o2_alfa[pos_i]
            aj, bj, cj = self.coefs_o1[l, pos_j]
            rho = bj * np.array([
                api,
                bpi,
                cpi,
                dpi,
                epi,
                fpi,
            ])
            valor = det * (rho * self.fatores_rho[:len(rho)]).sum(axis=0)
        else:
            raise NotImplementedError(f"Nao implementado calculo da derivada com teste ordem {ordem_i} e tentativa ordem {ordem_j}")

        return valor

    def procedimento_derivy(self, l, pos_i, pos_j, x, y, ordem_i, ordem_j):
        '''Procedimento elementar que quando aplicado a um vetor de valores nodais de f(x), retorna a integral de dNj(x)/dy*vi'''
        det = self.dets_lin[l]
        if ordem_i == 1 and ordem_j == 2:
            ##Esse eh o caso relevante para a equacao da divergencia
            api, bpi, cpi = self.coefs_o1_alfa[pos_i]
            aj, bj, cj, dj, ej, fj = self.coefs_o2[l, pos_j]
            rho = np.array([
                api * cj,
                api * (2 * ej * y[1] + fj * x[1]) + bpi * cj,
                api * (2 * ej * y[2] + fj * x[2]) + cpi * cj,
                bpi * (2 * ej * y[1] + fj * x[1]),
                cpi * (2 * ej * y[2] + fj * x[2]),
                bpi * (2 * ej * y[2] + fj * x[2]) + cpi * (2 * ej * y[1] + fj * x[1]),
            ])
            valor = det * (rho * self.fatores_rho[:len(rho)]).sum(axis=0)
        elif ordem_i == 2 and ordem_j == 1:
            api, bpi, cpi, dpi, epi, fpi = self.coefs_o2_alfa[pos_i]
            aj, bj, cj = self.coefs_o1[l, pos_j]
            rho = cj * np.array([
                api,
                bpi,
                cpi,
                dpi,
                epi,
                fpi,
            ])
            valor = det * (rho * self.fatores_rho[:len(rho)]).sum(axis=0)
        else:
            raise NotImplementedError(f"Nao implementado calculo da derivada com teste ordem {ordem_i} e tentativa ordem {ordem_j}")

        return valor

    def procedimento_integracao_num(self, l, pos_i, pos_j, x, y, ordem):
        x = np.vstack((x, y)).T
        expressao = lambda r: self.N_rel(pos_i, l, r, ordem) * self.N_rel(pos_j, l, r, ordem)
        return np.average([expressao(x_i) for x_i in x], axis=0) * self.dets_lin[l] / 2

    def procedimento_laplaciano_num(self, l, pos_i, pos_j, x, y, ordem):
        x = np.vstack((x, y)).T
        expressao = lambda r: - np.dot(self.grad_N_rel(pos_i, l, r, ordem), self.grad_N_rel(pos_j, l, r, ordem))
        return np.average([expressao(x_i) for x_i in x], axis=0) * self.dets_lin[l] / 2

    def monta_matriz_dirichlet(self, contornos_dirichlet=[], ordem=1):
        '''Monta as linhas da matriz da forma bilinear que correspondem a condicao de contorno de dirichlet e as entradas correspondentes do vetor no lado esquerdo
        :param contornos_dirichlet: lista de tuplas contendo, cada uma: (vetor de nos com condicao de contorno de Dirichlet, funcao u=f(x) que define a condicao de contorno)
        :param ordem: ordem da malha (1 ou 2)
        :return A: ssp.coo_matrix. Matriz esparsa apenas com as entradas correspondentes a condicao de Dirichlet
        return b: np.ndarray. Vetor com as entradas correspondentes a condicao de Dirichlet'''
        ##Definicoes iniciais
        if ordem == 1:
            elementos = self.elementos_o1
            nos = self.nos_o1
        elif ordem == 2:
            elementos = self.elementos
            nos = self.nos
        else:
            raise ValueError(f"Elementos de ordem {ordem} nao sao suportados")
        linhas, colunas, valores = [], [], []
        b = np.zeros(len(nos), dtype=np.float64)
        ###condicoes de dirichlet:
        for (cont, funcao) in contornos_dirichlet:
            for i in cont:
                ind_i = np.nonzero(nos == i)[0][0]  # indice do ponto i no vetor de nos (se ordem=1, o vetor de nos usado eh menor que o vetor total, se a malha tiver ordem 2)
                # A[ind_i, ind_i] = 1
                linhas.append(ind_i)
                colunas.append(ind_i)
                valores.append(1)
                b[ind_i] = funcao(self.x_nos[i])
        ##Geramos a matriz esparsa com base nos arrays de linhas, colunas e valores
        A = ssp.coo_matrix((valores, (linhas, colunas)), shape=(len(nos), len(nos)), dtype=np.float64)
        return A, b

    def monta_matriz_old(self, procedimento, contornos_dirichlet=[], ordem=1):
        '''Monta a matriz A que, aplicada ao vetor de valores de u ou p, produz a integral de uma determinada grandeza (que depende do procedimento).
        Atua em todos os pontos nos quais nao ha condicao de dirichlet, mas nao realiza qualquer tipo de calculo no contorno
        Ex: montar matriz que corresponde a forma variacional da equacao de Laplace
        :param contornos_dirichlet: lista de tuplas contendo, cada uma: (vetor de nos com condicao de contorno de Dirichlet, funcao u=f(x) que define a condicao de contorno)
        '''
        if ordem == 1:
            elementos = self.elementos_o1
            nos = self.nos_o1
        elif ordem == 2:
            elementos = self.elementos
            nos = self.nos
        else:
            raise ValueError(f"Elementos de ordem {ordem} nao sao suportados")
        linhas, colunas, valores = [], [], []
        pontos_dirichlet = np.concatenate([contornos_dirichlet[i][0] for i in range(len(contornos_dirichlet))])  # lista de todos os pontos com condicao de dirichlet
        pontos_sem_dirichlet = np.setdiff1d(nos, pontos_dirichlet)  # lista de todos os pontos sem condicao de dirichlet
        ##TODO paralelizar loop principal abaixo
        for i in pontos_sem_dirichlet:  # aqui, varremos todos os pontos nos quais a funcao teste nao eh identicamente nula
            elementos_i = np.where(elementos == i)[0]  # lista de elementos que contem o ponto i
            for l in elementos_i:  # varremos cada elemento no qual N_i != 0
                pos_i = np.nonzero(elementos[l] == i)[0][0]  # posicao do ponto i no elemento l
                if ordem == 1:
                    ind_i = np.nonzero(nos == i)[0][0]  # indice do ponto i no vetor de nos
                elif ordem == 2:
                    ind_i = i
                for j in elementos[l]:  # varremos os pontos do elemento l
                    x_abs, y_abs, z_abs = self.x_nos[elementos[l]].T
                    x, y = x_abs - x_abs[0], y_abs - y_abs[0]
                    pos_j = np.nonzero(elementos[l] == j)[0][0]  # posicao do ponto j no elemento l
                    if ordem == 1:
                        ind_j = np.nonzero(nos == j)[0][0]
                    elif ordem == 2:
                        ind_j = j
                    valor = procedimento(l, pos_i, pos_j, x, y, ordem)
                    linhas.append(ind_i)
                    colunas.append(ind_j)
                    valores.append(valor)

        ##Geramos a matriz esparsa com base nos arrays de linhas, colunas e valores
        A = ssp.coo_matrix((valores, (linhas, colunas)), shape=(len(nos), len(nos)), dtype=np.float64)
        # A = A.tobsr()
        return A

    def monta_matriz(self, procedimento, contornos_dirichlet=[], ordem_teste=1, ordem_tentativa=1, ordem=None):
        '''Monta a matriz A que, aplicada ao vetor de valores de u ou p, produz a integral de uma determinada grandeza (que depende do procedimento).
        Atua em todos os pontos nos quais nao ha condicao de dirichlet, mas nao realiza qualquer tipo de calculo no contorno
        Exige a definicao da ordem da funcao de interpolacao para a funcao teste (v ou q) e para a funcao tentativa (u ou p)
        Se n_v eh o numero de nos da funcao teste, e n_u da funcao tentativa, a matriz A tera dimensao n_v x n_u, pois cada entrada da matriz corresponde a um par (v_i, u_j)'''
        if not ordem is None:  # Nesse caso, a ordem eh a mesma para teste e tentativa, e os parametros ordem_teste e ordem_tentativa sao ignorados
            ordem_teste = ordem
            ordem_tentativa = ordem
        if ordem_teste == 1:
            elementos_teste = self.elementos_o1
            nos_teste = self.nos_o1
        elif ordem_teste == 2:
            elementos_teste = self.elementos
            nos_teste = self.nos
        else:
            raise ValueError(f"Elementos de ordem {ordem_teste} nao sao suportados")
        if ordem_tentativa == 1:
            elementos_tentativa = self.elementos_o1
            nos_tentativa = self.nos_o1
        elif ordem_tentativa == 2:
            elementos_tentativa = self.elementos
            nos_tentativa = self.nos
        else:
            raise ValueError(f"Elementos de ordem {ordem_tentativa} nao sao suportados")
        linhas, colunas, valores = [], [], []
        ###Definindo os pontos com condicao de dirichlet, onde a funcao tentativa eh nula
        pontos_dirichlet = np.concatenate([contornos_dirichlet[i][0] for i in range(len(contornos_dirichlet))])  # lista de todos os pontos com condicao de dirichlet
        pontos_sem_dirichlet = np.setdiff1d(nos_teste, pontos_dirichlet)  # lista de nos da funcao teste sem condição de dirichlet
        for i in pontos_sem_dirichlet:  # aqui, varremos todos os pontos nos da funcao TESTE
            elementos_i = np.where(elementos_teste == i)[0]  # lista de elementos que contem o ponto i
            for l in elementos_i:  # varremos cada elemento no qual N_i != 0
                pos_i = np.nonzero(elementos_teste[l] == i)[0][0]  # posicao do ponto i no elemento l
                if ordem_teste == 1:
                    ind_i = np.nonzero(nos_teste == i)[0][0]  # indice do ponto i no vetor de nos
                elif ordem_teste == 2:
                    ind_i = i
                for j in elementos_tentativa[l]:  # varremos os pontos da funcao TENTATIVA no elemento l
                    x_abs, y_abs, z_abs = self.x_nos[elementos_tentativa[l]].T
                    x, y = x_abs - x_abs[0], y_abs - y_abs[0]
                    pos_j = np.nonzero(elementos_tentativa[l] == j)[0][0]  # posicao do ponto j no elemento l
                    if ordem_tentativa == 1:
                        ind_j = np.nonzero(nos_tentativa == j)[0][0]
                    elif ordem_tentativa == 2:
                        ind_j = j
                    valor = procedimento(l, pos_i, pos_j, x, y, ordem_teste, ordem_tentativa)
                    linhas.append(ind_i)
                    colunas.append(ind_j)
                    valores.append(valor)

        A = ssp.coo_matrix((valores, (linhas, colunas)), shape=(len(nos_teste), len(nos_tentativa)), dtype=np.float64)
        return A

    def monta_tensor_convectivo(self, ordem=2):
        '''Monta o tensor correspondente a integral do termo de conveccao Ni*Nj*dNk/dx'''
        ##TODO lidar com condicoes de dirichlet
        if ordem==2:
            p=6
            elementos=self.elementos
        else: raise NotImplementedError(f"Ainda nao implementado para ordem {ordem}")
        ###Definicao da expressao dos termos q,r,s que definem dN/dx= q + r*alfa + s*beta
        calc_qrs={"x":lambda a,b,c,d,e,f, x, y: (b,2*d*x[1]+f*y[1], 2*d*x[2]+f*y[2]),
                  "y": lambda a,b,c,d,e,f, x, y: (c,2*e*y[1]+f*x[1], 2*e*y[2]+f*x[2])}
        n_elem=len(self.elementos)
        tensores= {"x":np.zeros(( p,n_elem, p, p)), "y":np.zeros(( p,n_elem, p, p))}
        #O tensor sera simetrico entre i e j, logo nao eh necessario fazer o loop para j>i
        x_elem=self.x_nos[elementos]
        for l in range(n_elem):
            x,y,z=(x_elem[l]-x_elem[l,0]).T
            det=self.dets_lin[l]
            for i in range(p):
                ai,bi,ci,di,ei,fi=self.coefs_o2_alfa[i]
                for j in range(6-i):
                    aj,bj,cj,dj,ej,fj=self.coefs_o2_alfa[j]
                    for k in range(6):
                        ak,bk,ck,dk,ek,fk=self.coefs_o2_alfa[k]
                        for direcao_derivada in ("x","y"):
                            q,r,s=calc_qrs[direcao_derivada](ak,bk,ck,dk,ek,fk,x,y)
                            rho=np.array([
                                ai * aj * q, #constante
                                ai * aj * r + ai * bj * q + aj * bi * q, #alfa
                                ai * bj * r + ai * dj * q + aj * bi * r + aj * di * q + bi * bj * q, #beta
                                ai * cj * s + ai * ej * q + aj * ci * s + aj * ei * q + ci * cj * q, #alfa²
                                ai * cj * s + ai * ej * q + aj * ci * s + aj * ei * q + ci * cj * q, #beta²
                                ai * bj * s + ai * cj * r + ai * fj * q + aj * bi * s + aj * ci * r + aj * fi * q + bi * cj * q + bj * ci * q, #alfa*beta
                                ai * dj * r + aj * di * r + bi * bj * r + bi * dj * q + bj * di * q, #alfa³
                                ai * ej * s + aj * ei * s + ci * cj * s + ci * ej * q + cj * ei * q, #beta³
                                ai * dj * s + ai * fj * r + aj * di * s + aj * fi * r + bi * bj * s + bi * cj * r + bi * fj * q + bj * ci * r + bj * fi * q + ci * dj * q + cj * di * q, #alfa²*beta
                                ai * ej * r + ai * fj * s + aj * ei * r + aj * fi * s + bi * cj * s + bi * ej * q + bj * ci * s + bj * ei * q + ci * cj * r + ci * fj * q + cj * fi * q, #alfa*beta²
                                bi * dj * r + bj * di * r + di * dj * q, #alfa⁴
                                ci * ej * s + cj * ei * s + ei * ej * q, #beta⁴
                                bi * dj * s + bi * fj * r + bj * di * s + bj * fi * r + ci * dj * r + cj * di * r + di * fj * q + dj * fi * q, #alfa³*beta
                                bi * ej * r + bi * fj * s + bj * ei * r + bj * fi * s + ci * dj * s + ci * fj * r + cj * di * s + cj * fi * r + di * ej * q + dj * ei * q + fi * fj * q, #alfa²*beta²
                                bi * ej * s + bj * ei * s + ci * ej * r + ci * fj * s + cj * ei * r + cj * fi * s + ei * fj * q + ej * fi * q, #alfa*beta³
                                di * dj * r, #alfa⁵
                                ei * ej * s, #beta⁵
                                di * dj * s + di * fj * r + dj * fi * r, #alfa⁴*beta
                                di * ej * r + di * fj * s + dj * ei * r + dj * fi * s + fi * fj * r, #alfa³*beta²
                                di * ej * s + dj * ei * s + ei * fj * r + ej * fi * r + fi * fj * s, #alfa²*beta³
                                ei * ej * r + ei * fj * s + ej * fi * s, #alfa*beta⁴
                            ])
                            integracao=det*(rho*self.fatores_rho[:len(rho)]).sum(axis=0)
                            tensores[direcao_derivada][i,l,j,k]=integracao
                            tensores[direcao_derivada][j,l,i,k]=integracao
        return tensores["x"], tensores["y"]






    def matriz_laplaciano_escalar(self, contornos_dirichlet=[], contornos_neumann=[], contornos_livres=[], ordem=1):
        A = self.monta_matriz(self.procedimento_laplaciano, contornos_dirichlet, ordem=ordem)
        return A

    def integrar_expressao(self, expressao, elemento, n):
        '''Integra numericamente uma expressao em um determinado elemento usando quadratura gaussiana'''
        # TODO implementar para elementos de ordem qualquer e com quadratura gaussiana de verdade
        # func_x=lambda alfa, beta: (1-alfa-beta)*self.x_nos[elemento,0,0]+alfa*self.x_nos[elemento,1,0]+beta*self.x_nos[elemento,2,0]
        # func_y=lambda alfa, beta: (1-alfa-beta)*self.x_nos[elemento,0,1]+alfa*self.x_nos[elemento,1,1]+beta*self.x_nos[elemento,2,1]
        # func_alfa=lambda ksi, eta: (1+ksi)/2
        # func_beta=lambda ksi, eta: (1-ksi)*(1+eta)/4
        # warnings.warn(NotImplementedError("Integracao numerica ainda nao implementada para elementos de ordem 2"))
        x = self.x_nos[self.elementos[elemento]]

        return np.average([expressao(x_i) for x_i in x], axis=0) * self.dets_lin[elemento] / 2

        pass

    def funcoes_forma(self, ordem=1):
        '''Calcula os coeficientes das funcoes de forma de cada no em cada elemento da malha'''
        if ordem == 1:
            elementos = self.elementos_o1
            nos = self.nos_o1
            coefs_a = np.zeros(shape=elementos.shape, dtype=np.float64)
            coefs_b = np.zeros(shape=elementos.shape, dtype=np.float64)
            coefs_c = np.zeros(shape=elementos.shape, dtype=np.float64)
            dets = np.zeros(len(elementos), dtype=np.float64)
            for n in range(len(elementos)):
                x_abs, y_abs, z_abs = self.x_nos[elementos[n]].T  # coordenadas absolutas dos nos do elemento n
                x = x_abs - x_abs[0]  # shiftando as coordenadas para serem relativas ao primeiro no, de modo a reduzir erros numericos
                y = y_abs - y_abs[0]
                matriz = np.array([[1., x[0], y[0]], [1., x[1], y[1]], [1., x[2], y[2]]])
                det_A = np.linalg.det(matriz)
                dets[n] = det_A
                for i in range(3):
                    j = (i + 1) % 3
                    k = (i + 2) % 3
                    coefs_a[n, i] = (x[j] * y[k] - x[k] * y[j]) / det_A
                    coefs_b[n, i] = (y[j] - y[k]) / det_A
                    coefs_c[n, i] = (x[k] - x[j]) / det_A
            return coefs_a, coefs_b, coefs_c, dets
        elif ordem == 2:
            elementos = self.elementos
            nos = self.nos
            coefs = np.zeros(shape=(len(elementos), 6, 6), dtype=np.float64)  # array N x 6 x 6 contendo todos os coeficientes da funcao N em cada no em cada elemento
            ##A funcao de forma eh da forma N(x,y)= a + bx + cy + dx² + ey² + fxy
            ##A matriz de coeficientes eh da forma [a, b, c, d, e, f]
            for n in range(len(elementos)):
                x_abs, y_abs, z_abs = self.x_nos[elementos[n]].T
                x = x_abs - x_abs[0]  # shiftando as coordenadas para serem relativas ao primeiro no, de modo a reduzir erros numericos
                y = y_abs - y_abs[0]
                matriz = np.vstack((np.ones(6), x, y, x ** 2, y ** 2, x * y)).T
                for i in range(6):
                    b = np.identity(6)[i]
                    coefs[n, i] = np.linalg.solve(matriz, b)
            return coefs
        else:
            raise ValueError(f"Elementos de ordem {ordem} nao sao suportados")

    def funcoes_forma_alfa(self, ordem=1):
        '''Calcula os coeficientes das funcoes de forma para cada elemento da malha, escrevendo a funcao de forma em funcao de alfa e beta (e.g. N=a+b*alfa+c*beta)
        Onde x= x0+ alfa*x1 + beta*x2, y=y0+ alfa*y1 + beta*y2
        Nesse caso, nao eh preciso um array de coeficientes para cada elemento, pois os coeficientes sao iguais em todos os elementos
        '''
        if ordem == 1:
            alfa = np.array([0, 1, 0])
            beta = np.array([0, 0, 1])
            n = 3  # numero de nos no elemento
            matriz = np.vstack([np.ones(n), alfa, beta]).T
        elif ordem == 2:
            alfa = np.array([0, 1, 0, .5, .5, 0])
            beta = np.array([0, 0, 1, 0, .5, .5])
            n = 6
            matriz = np.vstack([np.ones(n), alfa, beta, alfa ** 2, beta ** 2, alfa * beta]).T
        else:
            raise ValueError(f"Elementos de ordem {ordem} nao sao suportados")
        coefs = np.zeros(shape=(n, n), dtype=np.float64)
        for i in range(n):
            b = np.identity(n)[i]
            coefs[i] = np.linalg.solve(matriz, b)
        return coefs

    def N(self, no, elemento, x, ordem=1):
        '''Calcula a funcao de forma N_no(x,y) no elemento dado, para um ponto (x,y) qualquer
        :param no: int. Indice do no do elemento em que se deseja calcular a funcao de forma
        :param elemento: int. Indice do elemento em que se deseja calcular a funcao de forma
        :param x: array_like N×2. Array de pares de coordenadas do ponto em que se deseja calcular a funcao de forma'''
        if len(x.shape) == 1:  # Nesse caso, foi passado um unico ponto (x,y)
            x, y = x[0:2]
        else:  # nesse caso, foi passado um array de pontos [(x1,y1),(x2,y2),...]
            x, y = x[:, 0], x[:, 1]
        if ordem == 1:
            elementos = self.elementos_o1
        elif ordem == 2:
            elementos = self.elementos
        else:
            raise NotImplementedError(f"Elementos de ordem {ordem} nao sao suportados")
        no_inicial = elementos[elemento][0]  # encontrando o no inicial do elemento, importante para determinar a coordenada relativa em que os coeficientes foram calculados
        x0, y0 = self.x_nos[no_inicial, 0:2]  # pegando o ponto zero de referencia das funcoes de forma
        x_rel, y_rel = x - x0, y - y0
        i = np.nonzero(elementos[elemento] == no)[0][0]  # Posicao relativa do no no elemento (0, 1 ou 2). Se o no nao pertence ao elemento, ocorre IndexError
        # i=elementos[elemento].index(no) #Posicao relativa do no no elemento (0, 1 ou 2). Se o no nao pertence ao elemento, ocorre ValueError
        if ordem == 1:
            return self.coefs_a_lin[elemento, i] + self.coefs_b_lin[elemento, i] * x_rel + self.coefs_c_lin[elemento, i] * y_rel
        elif ordem == 2:
            coefs = self.coefs_o2[elemento, i]
            return coefs[0] + coefs[1] * x_rel + coefs[2] * y_rel + coefs[3] * x_rel ** 2 + coefs[4] * y_rel ** 2 + coefs[5] * x_rel * y_rel

    def N_rel(self, pos_i, elemento, x, ordem):
        '''Calcula o valor de N a partir da posicao x relativa ao primeiro no  e do indice relativo do no na funcao'''
        if len(x.shape) == 1:  # Nesse caso, foi passado um unico ponto (x,y)
            x, y = x[0:2]
        else:  # nesse caso, foi passado um array de pontos [(x1,y1),(x2,y2),...]
            x, y = x[:, 0], x[:, 1]
        if ordem == 1:
            return self.coefs_a_lin[elemento, pos_i] + self.coefs_b_lin[elemento, pos_i] * x + self.coefs_c_lin[elemento, pos_i] * y
        elif ordem == 2:
            coefs = self.coefs_o2[elemento, pos_i]
            return coefs[0] + coefs[1] * x + coefs[2] * y + coefs[3] * x ** 2 + coefs[4] * y ** 2 + coefs[5] * x * y

    def grad_N(self, no, elemento, x, ordem=1):
        '''Calcula o gradiente da funcao de forma N_no(x,y) no elemento dado, para um ponto (x,y) qualquer
        :param no: int. Indice do no do elemento em que se deseja calcular a funcao de forma
        :param elemento: int. Indice do elemento em que se deseja calcular a funcao de forma
        :param x: array_like N×2. Array de pares de coordenadas do ponto em que se deseja calcular a funcao de forma'''
        # if len(x.shape) == 1:  # Nesse caso, foi passado um unico ponto (x,y)
        #     x, y = x[0:2]
        # else:  # nesse caso, foi passado um array de pontos [(x1,y1),(x2,y2),...]
        #     x, y = x[:, 0], x[:, 1]
        if ordem == 1:
            elementos = self.elementos_o1
        elif ordem == 2:
            elementos = self.elementos
        else:
            raise NotImplementedError(f"Elementos de ordem {ordem} nao sao suportados")
        no_inicial = elementos[elemento][0]  # encontrando o no inicial do elemento, importante para determinar a coordenada relativa em que os coeficientes foram calculados
        # x0, y0 = self.x_nos[no_inicial, 0:2]  # pegando o ponto zero de referencia das funcoes de forma
        # x_rel, y_rel = x - x0, y - y0
        vetor_x0 = self.x_nos[no_inicial, 0:2]
        vetor_x_rel = x[:, :2] - vetor_x0  # pegando o ponto zero de referencia das funcoes de forma
        pos_i = np.nonzero(elementos[elemento] == no)[0][0]  # Posicao relativa do no no elemento (0, 1 ou 2). Se o no nao pertence ao elemento, ocorre IndexError
        # i=elementos[elemento].index(no) #Posicao relativa do no no elemento (0, 1 ou 2). Se o no nao pertence ao elemento, ocorre ValueError
        return self.grad_N_rel(pos_i, elemento, vetor_x_rel, ordem)
        # if ordem == 1:
        #     return np.stack((self.coefs_b_lin[elemento, i], self.coefs_c_lin[elemento, i])).T
        # elif ordem == 2:
        #     coefs = self.coefs_o2[elemento, i]
        #     return np.stack((coefs[1] + 2 * coefs[3] * x_rel + coefs[5] * y_rel, coefs[2] + 2 * coefs[4] * y_rel + coefs[5] * x_rel)).T

    def grad_N_rel(self, pos_i, elemento, x, ordem):
        '''Calcula o gradiente de N a partir da posicao x relativa ao primeiro no  e do indice relativo do no na funcao'''
        if len(x.shape) == 1:  # Nesse caso, foi passado um unico ponto (x,y)
            x, y = x[0:2]
        else:  # nesse caso, foi passado um array de pontos [(x1,y1),(x2,y2),...]
            x, y = x[:, 0], x[:, 1]
        if ordem == 1:
            return np.stack((self.coefs_b_lin[elemento, pos_i], self.coefs_c_lin[elemento, pos_i])).T
        elif ordem == 2:
            coefs = self.coefs_o2[elemento, pos_i]
            return np.stack((coefs[1] + 2 * coefs[3] * x + coefs[5] * y, coefs[2] + 2 * coefs[4] * y + coefs[5] * x)).T

    def escoamento_IPCS_Stokes(self, T=10., dt=0.1, ux_dirichlet=[], uy_dirichlet=[], p_dirichlet=[], Re=1, solucao_analitica=None, regiao_analitica=None, conveccao=False):
        '''Resolve um escoamento pelo metodo de desacoplamento de velocidade e pressao descrito em (Goda, 1978)
        Num primeiro momento, considera-se que as condicoes de contorno sao todas Dirichlet ou von Neumann homogeneo, entao as integrais no contorno sao desconsideradas
        Supoe-se que os pontos com condicao de dirchlet para ux sao os mesmos de uy, mas o valor da condicao de dirichlet em si pode ser diferente
        :param T: tempo total do escoamento
        :param dt: medida do passo de tempo a cada iteracao
        :param solucao_analitica: func. Solucao analitica do caso estacionario, se houver. Deve receber como argumento um array de pontos (x,y,z) e retornar um array de valores de u
        :param conveccao: bool. Se True, considera a conveccao na equacao de Navier-Stokes. Se False, supoe que o termo convectivo eh desprezivel, caindo na equacao de Stokes
        '''
        ##Definindo a estrutura da matriz de solucao
        n = len(self.nos)
        k = len(self.nos_o1)
        m = 2 * n + k  # duas dimensoes de velocidade (ordem 2) e uma de pressao (ordem 1)

        ##Inicializando os vetores de solucao
        u_n = np.zeros((len(self.nos), 2), dtype=np.float64)
        p_n = np.zeros(self.nos_o1.shape, dtype=np.float64)  # a ordem dos elementos da pressao deve ser menor que da velocidade

        ##Aplicando condicoes de dirichlet nos valores iniciais
        for (cont, funcao) in ux_dirichlet:
            for no in cont:
                u_n[no, 0] = funcao(self.x_nos[no])
        for (cont, funcao) in uy_dirichlet:
            for no in cont:
                u_n[no, 1] = funcao(self.x_nos[no])
        for (cont, funcao) in p_dirichlet:
            for no in cont:
                p_n[no] = funcao(self.x_nos[no])

        print(u"Montando as matrizes a serem usados pelo Método de Elementos Finitos")
        t1 = time.process_time()
        ##Montando as matrizes principais fora do loop
        mat_lap_o1 = self.matriz_laplaciano_escalar(contornos_dirichlet=p_dirichlet, ordem=1)
        mat_lap_o2 = self.matriz_laplaciano_escalar(contornos_dirichlet=ux_dirichlet, ordem=2)
        mat_integracao_o1 = self.monta_matriz(procedimento=self.procedimento_integracao_simples, contornos_dirichlet=p_dirichlet, ordem=1)
        mat_integracao_o2 = self.monta_matriz(procedimento=self.procedimento_integracao_simples, contornos_dirichlet=ux_dirichlet, ordem=2)
        mat_gradp_x = self.monta_matriz(procedimento=self.procedimento_derivx, contornos_dirichlet=ux_dirichlet, ordem_teste=2, ordem_tentativa=1)
        mat_gradp_y = self.monta_matriz(procedimento=self.procedimento_derivy, contornos_dirichlet=ux_dirichlet, ordem_teste=2, ordem_tentativa=1)
        mat_gradu_x = self.monta_matriz(procedimento=self.procedimento_derivx, contornos_dirichlet=p_dirichlet, ordem_teste=1, ordem_tentativa=2)
        mat_gradu_y = self.monta_matriz(procedimento=self.procedimento_derivy, contornos_dirichlet=p_dirichlet, ordem_teste=1, ordem_tentativa=2)
        ##u_ast
        matriz_bloco1 = mat_integracao_o2 / dt - mat_lap_o2 / Re
        A_dirich_ux, b_dirich_ux = self.monta_matriz_dirichlet(ux_dirichlet, ordem=2)
        A_dirich_uy, b_dirich_uy = self.monta_matriz_dirichlet(uy_dirichlet, ordem=2)
        nos_dirich_ux=np.concatenate([cont for (cont, funcao) in ux_dirichlet])
        nos_dirich_uy=np.concatenate([cont for (cont, funcao) in uy_dirichlet])
        A_u_ast = ssp.bmat([[matriz_bloco1 + A_dirich_ux, None], [None, matriz_bloco1 + A_dirich_uy]], format="csr")
        ##p_ast
        A_dirich_p, b_dirich_p = self.monta_matriz_dirichlet(p_dirichlet, ordem=1)
        b_dirich_p *= 0  # Como p_ast eh apenas a diferenca entre p_n+1 e p_n, a condicao de dirichlet para p_n+1 eh a mesma que para p_n
        vetor_dirich_u = np.concatenate((b_dirich_ux, b_dirich_uy))
        if conveccao:
            D_x, D_y=self.monta_tensor_convectivo(ordem=2) ##tensore relevantes para o termo convectivo derivado em x e y, respectivamente

        A_p = mat_lap_o1 + A_dirich_p
        ##u
        A_ux = mat_integracao_o2 / dt + A_dirich_ux
        A_uy = mat_integracao_o2 / dt + A_dirich_uy
        A_u = ssp.bmat([[A_ux, None], [None, A_uy]], format="csr")
        t2 = time.process_time()
        print(f"Tempo de montagem das matrizes: {t2 - t1:.2f} s")

        resultados = {}  # Dicionario contendo os resultados da simulacao para alguns passos de tempo

        tempos = np.arange(0, T + dt, dt)
        for t in tempos:
            print(f"Resolvendo para t={t}")
            t1 = time.process_time()
            ##Calculando u*
            # A matriz de solucao tem formato (2px2p), pois diz respeito apenas a velocidade





            vetor_un = np.concatenate(((mat_integracao_o2) @ u_n[:, 0], (mat_integracao_o2) @ u_n[:, 1]))
            vetor_gradp = np.concatenate((mat_gradp_x @ p_n, mat_gradp_y @ p_n))
            if conveccao:
                ux_elementos=u_n[:,0][self.elementos]
                uy_elementos=u_n[:,1][self.elementos]
                produtos_uxuy = produto_cartesiano_nodais(ux_elementos, uy_elementos, ordem=2)
                produtos_uxux = produto_cartesiano_nodais(ux_elementos, ux_elementos, ordem=2)
                produtos_uyuy = produto_cartesiano_nodais(uy_elementos, uy_elementos, ordem=2)
                ududx=calcula_termo_convectivo(produtos_uxux, D_x, self.pertencimento, nos_dirichlet=nos_dirich_ux)
                vdudy=calcula_termo_convectivo(produtos_uxuy, D_y, self.pertencimento, nos_dirichlet=nos_dirich_ux)
                termo_convectivo_x=ududx+vdudy
                udvdx=calcula_termo_convectivo(produtos_uxuy, D_x, self.pertencimento, nos_dirichlet=nos_dirich_uy)
                vdvdy=calcula_termo_convectivo(produtos_uyuy, D_y, self.pertencimento, nos_dirichlet=nos_dirich_uy)
                termo_convectivo_y=udvdx+vdvdy
                vetor_convectivo=np.concatenate((termo_convectivo_x, termo_convectivo_y))
                u_ast = ssp.linalg.spsolve(A_u_ast, vetor_un / dt - vetor_gradp - vetor_convectivo + vetor_dirich_u)
            else:
                u_ast = ssp.linalg.spsolve(A_u_ast, vetor_un / dt - vetor_gradp + vetor_dirich_u)
            u_ast = u_ast.reshape((2, len(self.nos))).T

            ##Calculando p*
            div_u_ast = mat_gradu_x @ u_ast[:, 0] + mat_gradu_y @ u_ast[:, 1]
            b_p = div_u_ast / dt + b_dirich_p
            p_ast = ssp.linalg.spsolve(A_p, b_p)

            ##Calculando p_n+1
            p = p_n + p_ast

            ##Calculando u_n+1
            vetor_u_ast = np.concatenate((mat_integracao_o2 @ u_ast[:, 0], mat_integracao_o2 @ u_ast[:, 1]))
            vetor_gradp = np.concatenate((mat_gradp_x @ p_ast, mat_gradp_y @ p_ast))
            b_u = vetor_u_ast / dt - vetor_gradp + vetor_dirich_u
            u = ssp.linalg.spsolve(A_u, b_u)
            u = u.reshape((2, len(self.nos))).T
            t2 = time.process_time()
            print(f"Tempo de resolucao: {t2 - t1:.2f} s")
            if not solucao_analitica is None:
                if not regiao_analitica is None:
                    pontos_an = self.nos[regiao_analitica(self.x_nos)]  # separa apenas os nos selecionados para avaliar a solucao analitica
                else:
                    pontos_an = self.nos
                u_an = solucao_analitica(Problema.x_nos[pontos_an])
                erro = u[pontos_an] - u_an
                print(f"Erro maximo: {np.max(np.abs(erro), axis=0)}")
                print(f"Erro RMS: {np.sqrt(np.average(erro ** 2, axis=0))}")
                print(f"Erro medio: {np.average(erro, axis=0)}")

            if np.isclose(t % 1, 0):
                resultados[t] = {"u": u, "u*": u_ast, "p": p, "p*": p_ast}
            u_n = u.copy()
            p_n = p.copy()
        return resultados

    def localiza_elemento(self, x, y):
        '''Dada um ponto, localiza em que elemento da malha ele se encontra'''

        pontos = self.x_nos[self.elementos_o1]
        x0, y0 = pontos[:, 0, :2].T  # vetor de pontos 0 de cada elemento
        x1, y1 = pontos[:, 1, :2].T
        x2, y2 = pontos[:, 2, :2].T
        pontos = None
        alfa = ((x - x0) * (y2 - y0) - (x2 - x0) * (y - y0)) / self.dets_lin  # vetor de coordenadas alfa do ponto em cada elemento
        beta = ((x1 - x0) * (y - y0) - (x - x0) * (y1 - y0)) / self.dets_lin  # vetor de coordenadas beta do ponto em cada elemento
        elemento = np.nonzero((alfa >= 0) & (beta >= 0) & (alfa + beta <= 1))[0][0]  # indice do primeiro elemento em que o ponto esta contido
        return elemento

    def interpola(self, x, u, ordem=1):
        '''Interpola a solucao u para um ponto (x,y) qualquer
        :param x: array_like N×2. Array de pares de coordenadas do ponto em que se deseja calcular a funcao de forma
        :param u: array_like N×2. Array de valores da solucao nos nos da malha
        :param ordem: int. Ordem do elemento em que se deseja interpolar a solucao
        '''

        elemento = self.localiza_elemento(x[0], x[1])
        if ordem == 1:
            elementos = self.elementos_o1
        elif ordem == 2:
            elementos = self.elementos
        else:
            raise ValueError(f"Elementos de ordem {ordem} nao sao suportados")
        nos = elementos[elemento]
        soma = sum([self.N(no, elemento, x, ordem=ordem) * u[no] for no in nos])
        return soma


if __name__ == "__main__":
    import pickle
    import AerofolioFino

    # tag_fis = {'esquerda': 1, 'direita': 2, 'superior': 3, 'inferior': 4, 'escoamento': 5}
    # nome_malha = "Malha/teste 5-1.msh"
    # ##Teste do produto cartesiano
    # matriz_teste1=np.random.random((42, 6))
    # matriz_teste2 = np.random.random((42, 6))
    # prod=produto_cartesiano_nodais(matriz_teste1, matriz_teste2, ordem=2)
    # for i in range(6):
    #     for j in range(6):
    #         if np.any(prod[i,j]!=matriz_teste1[:,i]*matriz_teste2[:,j]):
    #             print(f"Erro no elemento ({i}, {j}")

    # nome_malha, tag_fis = Malha.malha_retangular("teste 5-1", 0.05, (5, 1))
    nome_malha, tag_fis=Malha.malha_quadrada("grosseira", 0.1)

    Problema = FEA(nome_malha, tag_fis)
    ##Teste da criacao do tensor convectivo
    zero_u = np.zeros(shape=len(Problema.nos), dtype=np.float64)
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
    p_dirichlet = [(Problema.nos_cont_o1["direita"], lambda x: 0.),
                   # (Problema.nos_cont_o1["esquerda"], lambda x: 1.),
                   ]
    regiao_analitica = lambda x: np.logical_and(x[:, 0] >= 2, x[:, 0] <4.9)
    solucao_analitica = lambda x: np.vstack([6 * x[:, 1] * (1 - x[:, 1]), np.zeros(len(x))]).T
    executa = True
    if executa:
        resultados = Problema.escoamento_IPCS_Stokes(ux_dirichlet=ux_dirichlet, uy_dirichlet=uy_dirichlet, p_dirichlet=p_dirichlet, T=10, dt=0.05, Re=1, solucao_analitica=solucao_analitica, regiao_analitica=regiao_analitica, conveccao=True)
        with open(os.path.join("Picles", "resultados Navier-Stokes.pkl"), "wb") as f:
            pickle.dump((Problema, resultados), f)
    else:
        with open(os.path.join("Picles", "resultados.pkl"), "rb") as f:
            Problema, resultados = pickle.load(f)


    def plotar_momento(Problema, resultados, t):
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


    def plotar_perfis(Problema, resultados, t):
        fig, eixo = plt.subplots()
        for x in np.arange(0, 5.001, 0.5):
            plotar_perfil(Problema, resultados, t, x, eixo)
        eixo.legend()
        return

    t0=time.process_time()
    plotar_perfis(Problema, resultados, 10)
    t1=time.process_time()
    print(f"Perfis plotados em {t1-t0:.4f} s")

    # plotar_momento(Problema, resultados, 3)
    plotar_momento(Problema, resultados, 10)

    # nome_malha = "Malha/teste.msh"
    # for ordem in (1, 2,):
    #     for n_teste in (1, 2, 3, 4):
    #         teste_laplace(nome_malha, tag_fis, ordem=ordem, n_teste=n_teste, plota=True, gauss=False)
    # teste_laplace(nome_malha, tag_fis, ordem=2, n_teste=2)
    # teste_laplace(nome_malha, tag_fis, ordem=1, n_teste=2)
    # teste_laplace(nome_malha, tag_fis, ordem=2, n_teste=1)
    # teste_laplace(nome_malha, tag_fis, ordem=1, n_teste=1)
    plt.show(block=False)
    plt.pause(10)
    plt.show(block=True)
    print("r")
    # aerofolio = AerofolioFino.AerofolioFinoNACA4([0.04, 0.4, 0.12], 0, 1)
    # nome_malha, tag_fis = Malha.malha_aerofolio(aerofolio, nome_modelo="4412 grosseiro", n_pontos_contorno=2)
    # nome_malha = 'Malha/4412 grosseiro.msh'
    # fea = FEA(nome_malha, tag_fis)
    # elem0 = fea.elementos[0]
    # x_nos = fea.x_nos[elem0]
    # plt.scatter(x_nos[:, 0], x_nos[:, 1])
    # N=[(lambda x: fea.N(elem0[i], 0, x, ordem=2)) for i in range(len(elem0))]
    # N0 = lambda x : fea.N(elem0[0], 0, x, ordem=2)
    # N1 = lambda x : fea.N(elem0[1], 0, x, ordem=2)
    # N2 = lambda x : fea.N(elem0[2], 0, x, ordem=2)
    # print(N0(x_nos[:, 0 :2]))
    # print(N1(x_nos[:, 0 :2]))
    # print(N2(x_nos[:, 0 :2]))
    # # print(N0((x_nos[1] + x_nos[2]) / 2))
    # # print(N1((x_nos[1] + x_nos[2]) / 2))
    # # print(N2((x_nos[1] + x_nos[2]) / 2))
    # for i in range(6):
    #     N=lambda x: fea.N(elem0[i], 0, x, ordem=2)
    #     print(N(x_nos[:, 0:2]))
    # print("?")
