import warnings

from mpi4py import MPI
from dolfinx import log as dlog
dlog.set_log_level(dlog.LogLevel(0))
from dolfinx import mesh as dmesh
import dolfinx.io as dio
from dolfinx import fem as dfem
from dolfinx import nls as dnls
import dolfinx
import numpy as np
import os
from petsc4py.PETSc import ScalarType
import ufl  # Unified Form Language. Linguagem para definicao de problemas de elementos finitos e forma fraca
import gmsh
from Definicoes import *

geo = gmsh.model.geo  # definindo um alias para o modulo de geometria do gmsh
import dolfinx.io.gmshio

# MPI.COMM_WORLD permite a paralelizacao de uma mesma malha entre processadores diferentes
n_pontos_contorno_padrao=1000


def malha_aerofolio(aerofolio, nome_modelo="modelo", n_pontos_contorno=n_pontos_contorno_padrao) :
    '''Gera uma malha no gmsh correspondendo a regiao em torno do aerofolio'''
    ##TODO implementar
    contornos = {"entrada" : 1, "saida" : 2, "superior" : 3, "inferior" : 4, }
    # n_pontos_contorno = 1000
    tag_fis = {}  # tags dos grupos fisicos
    af_tamanho = 1 / n_pontos_contorno
    tamanho = 0.1
    ##Inicializando o gmsh
    gmsh.initialize()
    gmsh.model.add(nome_modelo)  # adiciona um modelo
    gmsh.model.set_current(nome_modelo)  # define o modelo atual
    geo.addPoint(-2, -1, 0, tamanho, tag=1)  # ponto inferior esquerdo
    geo.addPoint(-2, 1, 0, tamanho, tag=2)  # ponto superior esquerdo
    geo.addPoint(3, -1, 0, tamanho, tag=3)  # ponto inferior direito
    geo.addPoint(3, 1, 0, tamanho, tag=4)  # ponto superior direito
    geo.add_line(1, 2, tag=contornos["entrada"])
    geo.add_line(3, 4, tag=contornos["saida"])
    geo.add_line(1, 3, tag=contornos["inferior"])
    geo.add_line(2, 4, tag=contornos["superior"])

    ###versao a vera com aerofolio
    ponto_inicial = geo.add_point(aerofolio.x_med(0), aerofolio.y_med(0), 0, af_tamanho)
    ponto_final = geo.add_point(aerofolio.x_med(1), aerofolio.y_med(1), 0, af_tamanho)
    pontos_sup = [ponto_inicial, ]
    pontos_inf = [ponto_inicial, ]
    af_sup = []
    af_inf = []
    for i in range(1, n_pontos_contorno) :
        ##TODO escrever usando x e y vertores numpy com precisao 64bit
        ##eta eh igual ao x da linha base do aerofolio
        eta = i / n_pontos_contorno
        pontos_sup.append(geo.add_point(aerofolio.x_sup(eta), aerofolio.y_sup(eta), 0, af_tamanho))
        pontos_inf.append(geo.add_point(aerofolio.x_inf(eta), aerofolio.y_inf(eta), 0, af_tamanho))
    pontos_sup.append(ponto_final)
    pontos_inf.append(ponto_final)
    for i in range(n_pontos_contorno) :
        af_sup.append(geo.add_line(pontos_sup[i], pontos_sup[i + 1]))
        af_inf.append(geo.add_line(pontos_inf[i], pontos_inf[i + 1]))
    af_inf_inverso = [-item for item in af_inf[: :-1]]
    contornos["af_superior"] = af_sup
    contornos["af_inferior"] = af_inf

    ##DEBUG DEBUG DEBUG
    # desenha_aerofolio(pontos_sup,pontos_inf)

    ###Definindo as superficies para simulacao
    geo.add_curve_loop(af_sup + af_inf_inverso, tag=2)  # superficie do aerofolio
    geo.add_curve_loop([-1, 4, 2, -3], tag=1)  # superficie externa
    geo.add_plane_surface([1, 2], tag=1)  # superficie do escoamento

    ##Criando grupos fisicos correspondendo a cada elemento da simulacao
    tag_fis["af"] = geo.add_physical_group(1, af_sup + af_inf)
    tag_fis["entrada"] = geo.add_physical_group(1, [contornos["entrada"]])
    tag_fis["saida"] = geo.add_physical_group(1, [contornos["saida"]])
    tag_fis["superior"] = geo.add_physical_group(1, [contornos["superior"]])
    tag_fis["inferior"] = geo.add_physical_group(1, [contornos["inferior"]])
    tag_fis["escoamento"] = geo.add_physical_group(2, [1])

    ###Sincronizar as modificacoes geometricas e gerar a malha
    geo.synchronize()  # necessario!
    gmsh.model.mesh.generate(2)  # gera a malha
    nome_arquivo = os.path.join("Malha", f"{nome_modelo}.msh")
    gmsh.write(nome_arquivo)  # salva o arquivo da malha
    ##Encerrando o gmsh
    gmsh.finalize()
    return nome_arquivo


def desenha_aerofolio(pontos_sup, pontos_inf) :
    eixo = plt.axes()
    geo.synchronize()
    lista_x_sup = []
    lista_y_sup = []
    lista_x_inf = []
    lista_y_inf = []
    for ponto in pontos_sup :
        x, y, z = gmsh.model.get_value(0, ponto, [])
        lista_x_sup.append(x)
        lista_y_sup.append(y)
        # plt.scatter(x,y, color="blue")
        # plt.text(x,y, ponto, color="blue")
    for ponto in pontos_inf :
        x, y, z = gmsh.model.get_value(0, ponto, [])
        lista_x_inf.append(x)
        lista_y_inf.append(y)
        # plt.scatter(x,y, color="red")
        # plt.text(x,y, ponto, color="red")
    plt.plot(lista_x_sup, lista_y_sup, color="blue")
    plt.plot(lista_x_inf, lista_y_inf, color="red")
    eixo.set_xlim(-0.05, 1.05)
    eixo.set_ylim(-0.55, 0.55)
    plt.show(block=False)


def calculo_aerofolio(aerofolio) :
    '''
    Calcula as propriedades aerodinamicas de um aerofolio
    :param aerofolio: objeto da classe AerofolioFino
    '''
    ##TODO implementar

    nome_arquivo = malha_aerofolio(aerofolio, aerofolio.nome)
    psi_h = solucao_escoamento(aerofolio, nome_arquivo)
    # TODO calcular as propriedades aerodinamicas
    print(psi_h)

class SolucaoEscoamento :
    def __init__(self, aerofolio, nome_malha, n_pontos_contorno=n_pontos_contorno_padrao, gerar_malha=True) :
        self.aerofolio=aerofolio
        self.n_pontos_contorno=n_pontos_contorno
        if gerar_malha :
            nome_malha=malha_aerofolio(aerofolio, nome_malha, n_pontos_contorno)

        self.resolve_escoamento(aerofolio, nome_malha)

    def resolve_escoamento(self, aerofolio, nome_malha, caso="inviscido"):
        '''Resolve o escoamento em torno de um aerofolio a partir da malha gerada pelo gmsh.
            Retorna a funcao potencial como um campo do dolfin
            :param aerofolio: objeto da classe AerofolioFino
            :param malha: nome do arquivo da malha gerada pelo gmsh
            :param caso: ("inviscido", "viscoso"). Define o tipo de escoamento
            '''
        ##TODO reescrever sem supor escoamento potencial (escoamentos potenciais sao irrotacionais, logo nao havera sustentacao)
        y_1, y_2 = -1., 1.
        x_1, x_2 = -2., 3.
        self.limites= [[x_1, y_1], [x_2, y_2]]
        U0 = aerofolio.U0
        alfa = aerofolio.alfa
        self.malha, self.cell_tags, self.facet_tags = dio.gmshio.read_from_msh(nome_malha, MPI.COMM_WORLD, rank=0, gdim=2)

        v_cg2 = ufl.VectorElement("Lagrange", self.malha.ufl_cell(), 2)  # elemento vetorial de Lagrange de ordem 2, ligado a velocidade
        q_cg1 = ufl.FiniteElement("Lagrange", self.malha.ufl_cell(), 1)  # elemento escalar de Lagrange de ordem 1, ligado a pressao
        V = dfem.FunctionSpace(self.malha, v_cg2)  # Espaco de funcao da velocidade
        Q = dfem.FunctionSpace(self.malha, q_cg1)  # Espaco de funcao da pressao
        self.espaco_V = V
        self.espaco_Q = Q

        if caso=="inviscido":

            p_0 = 0.
            p_1=p_0
            cond_p_entrada = lambda x: p_0 + x[0] * 0.  # define-se p=p_0 na entrada
            # cond_p_saida = lambda x: p_1 + x[0] * 0.  #idem para p_1 na saida
            # p_lateral = lambda x: p_0 + x[0]*0.   #pressao e velocidade constantesa nas laterais
            u_0=np.array([U0,0.]) #velocidade de entrada vetorial
            u_0_m=dfem.Constant(self.malha, u_0) #velocidade de entrada como campo do dolfin
            cond_u_entrada=lambda x: (u_0 + x[:2].T * 0.).T  # define-se u=u_0 na entrada
            cond_u_lateral=lambda x: (u_0 + x[:2].T* 0.).T  # u nas laterais tem mesma velocidade da entrada nao perturbada, por estar afastado do obstaculo
            cond_u_aerofolio=lambda x: (u_0 + x[:2].T * 0.).T  # Tentativa conceitualmente incorreta com velocidade nula no aerofolio
            #Na saida, fazemos du/dx=0
            p_in = dfem.Function(Q)
            # p_out = dfem.Function(Q)
            u_in = dfem.Function(V)
            # u_out = dfem.Function(V)
            u_lateral = dfem.Function(V)
            u_aerofolio=dfem.Function(V)
            p_in.interpolate(cond_p_entrada)
            # p_out.interpolate(cond_p_saida)
            u_in.interpolate(cond_u_entrada)
            # u_out.interpolate(cond_u_saida)
            u_lateral.interpolate(cond_u_lateral)
            u_aerofolio.interpolate(cond_u_aerofolio)
        else:
            raise NotImplementedError("Apenas escoamentos inviscidos sao aceitos no momento!")
        tdim = self.malha.topology.dim  # dimensao do espaco (no caso, 2D)
        fdim = tdim - 1  # dimensao do contorno (no caso, 1D)
        boundary_facets = dmesh.exterior_facet_indices(self.malha.topology)  # indices dos segmentos dos contornos
        boundary_dofs = dfem.locate_dofs_topological(V, fdim, boundary_facets)  # indices dos graus de liberdade dos segmentos dos contornos
        contorno_entrada = dfem.locate_dofs_geometrical(V, lambda x : np.isclose(x[0], x_1))
        contorno_saida = dfem.locate_dofs_geometrical(V, lambda x : np.isclose(x[0], x_2))
        contorno_superior = dfem.locate_dofs_geometrical(V, lambda x : np.isclose(x[1], y_2))
        contorno_inferior = dfem.locate_dofs_geometrical(V, lambda x : np.isclose(x[1], y_1))
        contornos_externos = np.concatenate([contorno_superior, contorno_inferior, contorno_entrada, contorno_saida])
        self.contorno_aerofolio = np.setdiff1d(boundary_dofs, contornos_externos)
        # TODO definir contorno do aerofolio geometricamente
        ### Condicoes de contorno de dirichlet para velocidade
        bc_entrada_V = dfem.dirichletbc(u_in, contorno_entrada)  # aplica a condicao de contorno de Dirichlet com valor u_in
        # bc_saida_V = dfem.dirichletbc(u_out, contorno_saida)  # aplica a condicao de contorno de Dirichlet com valor u_out
        bc_superior_V = dfem.dirichletbc(u_lateral, contorno_superior)  # aplica a condicao de contorno de Dirichlet com valor u_lateral
        bc_inferior_V = dfem.dirichletbc(u_lateral, contorno_inferior)  # aplica a condicao de contorno de Dirichlet com valor u_lateral
        bc_aaerofolio_V = dfem.dirichletbc(u_aerofolio, self.contorno_aerofolio)  # aplica a condicao de contorno de Dirichlet com valor u_lateral
        condicoes_V=[bc_entrada_V, bc_superior_V, bc_inferior_V, bc_aaerofolio_V]
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)

        ##Condicoes de contorno de dirichlet para a pressao
        bc_entrada_p = dfem.dirichletbc(p_in, contorno_entrada)  # aplica a condicao de contorno de Dirichlet com valor p_in
        condicoes_Q=[bc_entrada_p]
        p=ufl.TrialFunction(Q)
        q=ufl.TestFunction(Q)

        n= ufl.FacetNormal(self.malha)

        #TODO refazer essa parte com u em vez de phi
        t_0=0
        t_1=10
        dt=0.1
        tempos=np.arange(t_0+dt,t_1+dt,dt)
        if caso=="inviscido":
            ##No passo 0, supomos p e u constantes na extensao do escoamento
            A_u_ast=ufl.dot(u,v)*ufl.dx
            L_u_ast=ufl.dot(u_0_m,v)*ufl.dx
            problema_inicial_u=dfem.petsc.LinearProblem(A_u_ast,L_u_ast,bcs=condicoes_V)
            u_inicial=problema_inicial_u.solve()
            p_inicial=dfem.Function(Q)
            p_inicial.interpolate(p_in)
            u_n=u_inicial
            p_n=p_inicial
            ##Escrevendo os resultados em arquivo

            # vtk_u=dio.VTKFile(self.malha.comm, "Saida/EF1/u.vtk", "w")
            # vtk_p=dio.VTKFile(self.malha.comm, "Saida/EF1/p.vtk", "w")
            # vtk_u.write_mesh(self.malha)
            # vtk_u.write_function(u_n, 0)
            # vtk_p.write_mesh(self.malha)
            # vtk_p.write_function(p_n, 0)

            for t in tempos:
                print(t)
                ##Passo 1: encontrar a velocidade tentativa u_ast (ast=asterisco)
                A_u_ast=1/dt*ufl.dot(u,v)*ufl.dx
                L_u_ast=1/dt*ufl.dot(u_n,v)*ufl.dx-ufl.dot(ufl.dot(u_n, ufl.nabla_grad(u_n)), v)*ufl.dx -ufl.dot(ufl.grad(p_n), v)*ufl.dx
                passo_1=dfem.petsc.LinearProblem(A_u_ast,L_u_ast,bcs=condicoes_V)
                u_ast= passo_1.solve()
                ##Passo 2: encontrar a pressao no passo n+1
                A_p=ufl.dot(ufl.grad(p),ufl.grad(q))*ufl.dx-q*ufl.dot(ufl.grad(p), n)*ufl.ds
                L_p=ufl.dot(ufl.grad(p_n),ufl.grad(q))*ufl.dx-q*ufl.dot(ufl.grad(p_n), n)*ufl.ds - 1/dt*q*ufl.div(u_ast)*ufl.dx
                passo_2=dfem.petsc.LinearProblem(A_p,L_p,bcs=condicoes_Q)
                p_novo=passo_2.solve()
                ##Passo 3: encontrar a velocidade no passo n+1
                A_u=1/dt*ufl.dot(u,v)*ufl.dx
                L_u=1/dt*ufl.dot(u_ast,v)*ufl.dx +ufl.dot(ufl.grad(p_n),v)*ufl.dx -ufl.dot(ufl.grad(p_novo), v)*ufl.dx
                passo_3=dfem.petsc.LinearProblem(A_u,L_u,bcs=condicoes_V)
                u_novo=passo_3.solve()

                u_n.interpolate(u_novo)
                p_n.interpolate(p_novo)
                ##Escrevendo os resultados em arquivo
                if t%0.5==0:
                    pass
                    # vtk_u.write_function(u_n, t)
                    # vtk_p.write_function(p_n, t)
        else:
            raise NotImplementedError("Apenas escoamentos inviscidos sao aceitos no momento!")

        # ##Definicao do solver nao linear por metodo de Newton
        # solver = dnls.NewtonSolver(MPI.COMM_WORLD, problema)
        # solver.convergence_criterion = "incremental"
        # solver.rtol = 1e-6
        # solver.report=True
        # n_it, convergencia = problema.solve()  # resolve o sistema pelo metodo nao-linear
        ##Armazenando os resultados como atributos da classe
        self.u=u_n
        self.p=p_n
        self.val_u=self.u.vector.array
        self.val_p=self.p.vector.array
        self.x=self.malha.geometry.x

    def interpola_solucao(self, n_pontos):
        '''Calcula a solucao da funcao potencial em uma malha regular de pontos'''
        #TODO implementar sem usar funcao potencial
        raise NotImplementedError("Funcao ainda esta configurada para usar escoamento potencial")
        x_1, y_1 = self.limites[0]
        x_2, y_2 = self.limites[1]
        self.malha_solucao=dmesh.create_rectangle(MPI.COMM_WORLD, [[x_1, y_1], [x_2, y_2]], [n_pontos, n_pontos], dmesh.CellType.quadrilateral)
        espaco_solucao=dfem.FunctionSpace(self.malha_solucao, ("CG", 1))
        self.phi_solucao=dfem.Function(espaco_solucao)
        self.phi_solucao.interpolate(self.phi)
        self.x_solucao=self.malha_solucao.geometry.x
        return self.x_solucao, self.phi_solucao.vector.array

    def ordena_contorno(self):
        '''Ordena os pontos do contorno do aerofolio em sentido anti-horario'''
        x_aerofolio=self.x[self.contorno_aerofolio]
        x, y =x_aerofolio[:,0], x_aerofolio[:,1]

        eta=np.arange(1,self.n_pontos_contorno)/self.n_pontos_contorno
        x_0,y_0=self.aerofolio.x_med(0), self.aerofolio.y_med(0)  #inicial
        x_f, y_f=self.aerofolio.x_med(1), self.aerofolio.y_med(1) #final
        x_sup, y_sup = self.aerofolio.x_sup(eta), self.aerofolio.y_sup(eta)
        x_inf, y_inf = self.aerofolio.x_inf(eta), self.aerofolio.y_inf(eta)
        pos_inf=np.array([x_inf,y_inf]).T
        pos_sup=np.array([x_sup,y_sup]).T
        caminho=np.concatenate([((x_0,y_0),),pos_inf,((x_f,y_f),),pos_sup[::-1]]) #comeca no ponto 0, faz o caminho por baixo e depois volta pór cima (sentido anti-horario)
        pontos=np.zeros(len(caminho),dtype=int)
        for i in range(len(pontos)):
            pontos[i]=np.argmin((x-caminho[i,0])**2+(y-caminho[i,1])**2) #indice de cada ponto do contorno na lista dos pontos de contorno
        self.indices_contorno=self.contorno_aerofolio[pontos] #indice de cada ponto na lista global de pontos da malha
        return self.indices_contorno

    def calcula_forcas(self):
        '''Calcula as forcas de sustentacao e arrasto e momento produzidas pela pressao'''
        #TODO implementar sem usar funcao potencial
        raise NotImplementedError("Funcao ainda esta configurada para usar escoamento potencial")
        lista_pontos=np.concatenate((self.indices_contorno, self.indices_contorno[0:1])) #da um loop completo, repetindo o ponto zero
        x, y, z=self.x[lista_pontos].T
        dx=x[1:]-x[:-1]
        dy=y[1:]-y[:-1] #o vetor entre um ponto e o seguinte eh (dx,dy)
        ds=np.sqrt(dx**2+dy**2) #comprimento de cada segmento
        dphi=self.val_phi[lista_pontos[1:]]-self.val_phi[lista_pontos[:-1]]
        ##pontos medios de cada segmento:
        x_med=(x[1:]+x[:-1])/2
        y_med=(y[1:]+y[:-1])/2
        u=dphi/ds #modulo da velocidade em cada ponto
        #vetor normal ao segmento entre pontos consecutivos
        normal=np.transpose([-dy/ds, dx/ds])
        ##Bernoulli: p/rho+1/2*U²=cte
        ##Todas as grandezas aqui retornadas sao divididas por unidade de massa (F_L/rho, F_D/rho, M/rho)
        ##Defina p/rho=0 no ponto 0
        pressao_total=0 + 1/2*u[0]**2
        pressao=pressao_total-1/2*u**2
        forcas=(pressao*ds*normal.T).T
        x_rel,y_rel = x_med-self.aerofolio.x_o, y_med-self.aerofolio.y_o #posicao do ponto central dos segmentos em relacao ao centro (quarto de corda) do aerofolio
        momentos=x_rel*forcas[:,1]-y_rel*forcas[:,0] #momento em relacao ao centro do aerofolio
        self.forca=forcas.sum(axis=0)
        self.F_D, self.F_L=self.forca #forcas de sustentacao e arrasto por unidade de massa especifica do fluido
        self.M=momentos.sum() #momento em relacao ao centro do aerofolio por unidade de massa especifica do fluido
        return self.F_L, self.F_D, self.M


    def linha_corrente(self, ponto_inicial):
        ##TODO tracar linha de corrente a partir da velocidade em cada ponto
        pass









def solucao_escoamento(aerofolio, nome_malha, n_pontos_contorno=n_pontos_contorno_padrao) :
    '''Resolve o escoamento em torno de um aerofolio a partir da malha gerada pelo gmsh.
    Retorna a funcao potencial como um campo do dolfin
    :param aerofolio: objeto da classe AerofolioFino
    :param malha: nome do arquivo da malha gerada pelo gmsh
    '''
    warnings.warn("Funcao depreciada. Use a classe SolucaoEscoamento", DeprecationWarning)
    y_1, y_2 = -1., 1.
    x_1, x_2 = -2., 3.
    U0 = aerofolio.U0
    alfa = aerofolio.alfa
    malha, cell_tags, facet_tags = dio.gmshio.read_from_msh(nome_malha, MPI.COMM_WORLD, rank=0, gdim=2)
    V = dfem.FunctionSpace(malha, ("CG", 1))  ##CG eh a familia de polinomios interpoladores de Lagrange, 1 eh a ordem do polinomio
    phi_0 = 0.
    phi_entrada = lambda x : phi_0 + x[0] * 0  # define-se psi=0 em y=0
    phi_saida = lambda x : phi_0 + U0 * (x_2 - x_1) + x[0] * 0
    phi_lateral = lambda x : phi_0 + U0 * x[0] - x_1
    u_in = dfem.Function(V)
    u_out = dfem.Function(V)
    u_lateral = dfem.Function(V)
    u_in.interpolate(phi_entrada)
    u_out.interpolate(phi_saida)
    u_lateral.interpolate(phi_lateral)
    tdim = malha.topology.dim  # dimensao do espaco (no caso, 2D)
    fdim = tdim - 1  # dimensao do contorno (no caso, 1D)
    boundary_facets = dmesh.exterior_facet_indices(malha.topology)  # indices dos segmentos dos contornos
    boundary_dofs = dfem.locate_dofs_topological(V, fdim, boundary_facets)  # indices dos graus de liberdade dos segmentos dos contornos
    contorno_entrada = dfem.locate_dofs_geometrical(V, lambda x : np.isclose(x[0], x_1))
    contorno_saida = dfem.locate_dofs_geometrical(V, lambda x : np.isclose(x[0], x_2))
    contorno_superior = dfem.locate_dofs_geometrical(V, lambda x : np.isclose(x[1], y_2))
    contorno_inferior = dfem.locate_dofs_geometrical(V, lambda x : np.isclose(x[1], y_1))
    contornos_externos = np.concatenate([contorno_superior, contorno_inferior, contorno_entrada, contorno_saida])
    contorno_aerofolio = np.setdiff1d(boundary_dofs, contornos_externos)


    bc_entrada = dfem.dirichletbc(u_in, contorno_entrada)  # aplica a condicao de contorno de Dirichlet com valor u_in
    bc_saida = dfem.dirichletbc(u_out, contorno_saida)  # aplica a condicao de contorno de Dirichlet com valor u_out
    bc_superior = dfem.dirichletbc(u_lateral, contorno_superior)  # aplica a condicao de contorno de Dirichlet com valor u_lateral
    bc_inferior = dfem.dirichletbc(u_lateral, contorno_inferior)  # aplica a condicao de contorno de Dirichlet com valor u_lateral

    phi = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.dot(ufl.grad(phi), ufl.grad(v)) * ufl.dx  # define a forma bilinear a(u,v) = \int_\Omega \del psi * \del v dx
    coord = ufl.SpatialCoordinate(malha)
    L = (coord[0] - coord[
        0]) * v * ufl.ds  # a forma linear nesse caso desaparece, pois as condicoes de contorno nas paredes, entrada e saida sao Dirichlet, e no aerofolio e von Neumann
    problema = dfem.petsc.LinearProblem(a, L, bcs=[bc_entrada, bc_saida, bc_superior,
                                                   bc_inferior], petsc_options={"ksp_type" : "preonly", "pc_type" : "lu"})  # fatoracao LU para solucao da matriz
    phi_h = problema.solve()  # resolve o sistema e retorna a funcao-solucao

    # ##Interpretando os dados
    # vetor = phi_h.vector.array
    # x = malha.geometry.x[:, 0]
    # y = malha.geometry.x[:, 1]
    # plt.figure()
    # plt.scatter(x, y, c=vetor)

    vals=np.empty(shape=len(malha.geometry.x), dtype=np.float64)
    eval=phi_h.eval(x=np.array([[x_1,y_1,0.],[x_2,y_2,0.]]), cells=[0,1])
    print(eval)
    outra_malha=dmesh.create_rectangle(MPI.COMM_WORLD, [[x_1, y_1], [x_2, y_2]], [10,10], dmesh.CellType.quadrilateral)
    Sol=dfem.FunctionSpace(outra_malha, ("CG", 1))
    phi_h2=dfem.Function(Sol)
    phi_h2.interpolate(phi_h)
    print(phi_h2.vector.array)
    return phi_h


def exemplo() :
    # from dolfinx.fem import FunctionSpace
    ##Vamos resolver a equacao de Poisson: \del² u = f
    ##O problema foi manufaturado com f=-6 para ter solucao exata u = 1+x²+2y²
    domain = dmesh.create_unit_square(MPI.COMM_WORLD, 8, 8, dmesh.CellType.triangle)
    V = dfem.FunctionSpace(domain, ("CG", 2))  ##CG eh a familia de polinomios interpoladores de Lagrange, 2 eh a ordem do polinomio
    uD = dfem.Function(V)
    funcao_contorno = lambda x : 1 + x[0] ** 2 + 2 * x[1] ** 2
    uD.interpolate(funcao_contorno)
    tdim = domain.topology.dim  # dimensao do espaco (no caso, 2D)
    fdim = tdim - 1  # dimensao do contorno (no caso, 1D)
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = dmesh.exterior_facet_indices(domain.topology)  # indices dos segmentos dos contornos
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)  # indices dos graus de liberdade dos segmentos dos contornos
    bc = fem.dirichletbc(uD, boundary_dofs)  # aplica a condicao de contorno de Dirichlet com valor uD
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    # funcao que coloca um escalar no type correto (e.g. np.float64)
    f = fem.Constant(domain, ScalarType(-6))  # define f como uma constante -6
    a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx  # define a forma bilinear a(u,v) = \int_\Omega \del u * \del v dx
    L = f * v * ufl.dx  # define a forma linear L(v) = \int_\Omega f*v dx
    # ufl.dot eh o produto matricial ou produto escalar entre 2 vetores
    problem = fem.petsc.LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type" : "preonly", "pc_type" : "lu"})  # fatoracao LU para solucao da matriz
    uh = problem.solve()  # resolve o sistema e retorna a funcao-solucao

    V2 = fem.FunctionSpace(domain, ("CG", 1))  # define outro dominio para avaliar o erro
    uex = fem.Function(V2)
    solucao_analitica = lambda x : 1 + x[0] ** 2 + 2 * x[1] ** 2
    uex.interpolate(solucao_analitica)
    L2_error = fem.form(ufl.inner(uh - uex, uh - uex) * ufl.dx)  # calcula o erro quadratico L2 entre a solucao numerica e a analitica
    error_local = fem.assemble_scalar(L2_error)  # calcula o valor escalar do erro no processo local
    error_global = np.sqrt(domain.comm.allreduce(error_local, op=MPI.SUM))  # soma os erros de todos os processos e tira a raiz quadrada
    error_max = np.max(np.abs(uD.x.array - uh.x.array))  # maximo erro em um grau de liberdade do contorno
    print(f"Erro L2: {error_global:.2e} \nErro maximo: {error_max:.2e}")

    ##Exportacao para o Paraview ##TODO debugar
    from dolfinx import io
    # with io.VTXWriter(domain.comm, "output.bp", [uh]) as vtx:
    #     vtx.write(0.0)
    with io.XDMFFile(domain.comm, os.path.join("Saida", "output.xdmf"), "w") as xdmf :
        xdmf.write_mesh(domain)
        xdmf.write_function(uh)

    ##Visualizacao de resultados
    import pyvista
    print(pyvista.global_theme.jupyter_backend)
    import dolfinx.plot
    pyvista.start_xvfb()
    topology, cell_types, geometry = dolfinx.plot.create_vtk_mesh(domain, domain.topology.dim)  # cria a malha em formato VTK para visualizacao no pyvista
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)  # cria o grid
    pyplotter = pyvista.Plotter()  # cria o objeto plotter
    pyplotter.add_mesh(grid, show_edges=True)  # adiciona a malha ao plotter
    pyplotter.view_xy()
    ##TODO debugar o pyvista
    if not pyvista.OFF_SCREEN :
        pyplotter.show()  # mostra a malha
    else :
        fig = pyplotter.screenshot(os.path.join("Saida", "malha.png"), )  # salva a malha em um arquivo png

    u_topology, u_cell_types, u_geometry = dolfinx.plot.create_vtk_mesh(V)
    u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
    u_grid.point_data["u"] = uh.x.array.real
    u_grid.set_active_scalars("u")
    u_plotter = pyvista.Plotter()
    u_plotter.add_mesh(u_grid, show_edges=True)
    u_plotter.view_xy()
    if not pyvista.OFF_SCREEN :
        u_plotter.show()

    warped = u_grid.warp_by_scalar()
    plotter2 = pyvista.Plotter()
    plotter2.add_mesh(warped, show_edges=True, show_scalar_bar=True)
    if not pyvista.OFF_SCREEN :
        plotter2.show()


if __name__ == "__main__" :
    import AerofolioFino

    aerofolio = AerofolioFino.AerofolioFinoNACA4([0.04, 0.4, 0.01], 0, 100)
    # nome_malha=malha_aerofolio(aerofolio, nome_modelo="NACA fino", n_pontos_contorno=1000)
    nome_malha = 'Malha/NACA4412.msh'
    solucao = SolucaoEscoamento(aerofolio, nome_malha, n_pontos_contorno=1000, gerar_malha=False)
    solucao.ordena_contorno()
    print(solucao.calcula_forcas())
    # x, phi = solucao.interpola_solucao(100)
    # plt.scatter(x[:, 0], x[:, 1], c=phi)
    # plt.figure()
    # indices=solucao.ordena_contorno()
    # plt.plot(solucao.x[indices,0], solucao.x[indices,1])
    plt.show(block=False)
    print("?")
