import warnings

from mpi4py import MPI
from dolfinx import log as dlog
dlog.set_log_level(dlog.LogLevel(0))
from dolfinx import mesh as dmesh
import dolfinx.io as dio
from dolfinx import fem as dfem
from dolfinx.fem import petsc as dpetsc
dppetsc=dpetsc.PETSc #estamos usando o pestsc do dolfin, em vez do petsc4py (isso eh um problema?)
##TODO consertar petsc
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

def exporta_valores(u, t, malha, path):
    '''Exporta os valores de u para um arquivo .csv'''

    return

def calculo_aerofolio(aerofolio) :
    '''
    Calcula as propriedades aerodinamicas de um aerofolio
    :param aerofolio: objeto da classe AerofolioFino
    '''
    ##TODO implementar

    nome_arquivo = malha_aerofolio(aerofolio, aerofolio.nome)


class SolucaoEscoamento :
    def __init__(self, aerofolio, nome_malha, viscosidade=1, n_pontos_contorno=n_pontos_contorno_padrao, gerar_malha=True, caso="inviscido") :
        self.aerofolio=aerofolio
        self.n_pontos_contorno=n_pontos_contorno
        self.viscosidade=viscosidade #viscosidade cinematica do fluido
        if gerar_malha :
            nome_malha=malha_aerofolio(aerofolio, nome_malha, n_pontos_contorno)

        self.resolve_escoamento(aerofolio, nome_malha, caso=caso)

    def resolve_escoamento(self, aerofolio, nome_malha, caso="inviscido"):
        '''Resolve o escoamento em torno de um aerofolio a partir da malha gerada pelo gmsh.
            Retorna a funcao potencial como um campo do dolfin
            :param aerofolio: objeto da classe AerofolioFino
            :param malha: nome do arquivo da malha gerada pelo gmsh
            :param caso: ("inviscido", "viscoso"). Define o tipo de escoamento
            '''
        y_1, y_2 = -1., 1.
        x_1, x_2 = -2., 3.
        self.limites= [[x_1, y_1], [x_2, y_2]]
        U0 = aerofolio.U0
        alfa = aerofolio.alfa
        self.malha, self.cell_tags, self.facet_tags = dio.gmshio.read_from_msh(nome_malha, MPI.COMM_WORLD, rank=0, gdim=2)


        v_cg2 = ufl.VectorElement("Lagrange", self.malha.ufl_cell(), 2, dim=2)  # elemento vetorial de Lagrange de ordem 2, ligado a velocidade
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
        elif caso=="viscoso":
            p_0 = 0.
            cond_p_entrada = lambda x: p_0 + x[0] * 0.  # define-se p=p_0 na entrada
            u_0 = np.array([U0, 0.])  # velocidade de entrada vetorial
            u_0_m = dfem.Constant(self.malha, u_0)  # velocidade de entrada como campo do dolfin
            cond_u_entrada = lambda x: (u_0 + x[:2].T * 0.).T  # define-se u=u_0 na entrada
            cond_u_lateral = lambda x: (u_0 + x[:2].T * 0.).T  # u nas laterais tem mesma velocidade da entrada nao perturbada, por estar afastado do obstaculo
            cond_u_aerofolio = lambda x: (u_0 + x[:2].T * 0.).T  # Tentativa conceitualmente incorreta com velocidade nula no aerofolio
            # Na saida, fazemos du/dx=0
            p_in = dfem.Function(Q)
            # p_out = dfem.Function(Q)
            u_in = dfem.Function(V)
            # u_out = dfem.Function(V)
            u_lateral = dfem.Function(V)
            u_aerofolio = dfem.Function(V)
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

        t_0=0
        t_1=10
        dt=0.1
        tempos=np.arange(t_0+dt,t_1+dt,dt)

        ##No passo 0, supomos p e u constantes na extensao do escoamento
        A_u_ast = ufl.dot(u, v) * ufl.dx
        L_u_ast = ufl.dot(u_0_m, v) * ufl.dx

        problema_inicial_u = dpetsc.LinearProblem(A_u_ast, L_u_ast, bcs=condicoes_V)
        u_inicial = problema_inicial_u.solve()
        p_inicial = dfem.Function(Q)
        p_inicial.interpolate(p_in)
        u_n = u_inicial
        p_n = p_inicial
        ##Variaveis de solucao
        u_ast = dfem.Function(V)
        p_novo = dfem.Function(Q)
        u_novo = dfem.Function(V)
        if caso=="inviscido":


            ##Definindo matriz e vetor que representam o problema

            #Passo 1
            A_u_ast = 1 / dt * ufl.dot(u, v) * ufl.dx
            L_u_ast = 1 / dt * ufl.dot(u_n, v) * ufl.dx - ufl.dot(ufl.dot(u_n, ufl.nabla_grad(u_n)), v) * ufl.dx - ufl.dot(ufl.grad(p_n), v) * ufl.dx

            #Passo 2
            A_p = ufl.dot(ufl.grad(p), ufl.grad(q)) * ufl.dx - q * ufl.dot(ufl.grad(p), n) * ufl.ds
            L_p = ufl.dot(ufl.grad(p_n), ufl.grad(q)) * ufl.dx - q * ufl.dot(ufl.grad(p_n), n) * ufl.ds - 1 / dt * q * ufl.div(u_ast) * ufl.dx

            #Passo 3
            A_u = 1 / dt * ufl.dot(u, v) * ufl.dx
            L_u = 1 / dt * ufl.dot(u_ast, v) * ufl.dx + ufl.dot(ufl.grad(p_n), v) * ufl.dx - ufl.dot(ufl.grad(p_novo), v) * ufl.dx

        elif caso=="viscoso":
            def epsilon(x):
                return ufl.sym(ufl.nabla_grad(x))
            def sigma(x,y):
                return 2*self.viscosidade*epsilon(x)-y*ufl.Identity(2)
            u_intermediario=0.5*(u_n+u) #velocidade intermediaria entre u_n anterior e u_ast

            F= 1 / dt * ufl.dot(u, v) * ufl.dx - 1 / dt * ufl.dot(u_n, v) * ufl.dx +ufl.dot(ufl.dot(u_n,ufl.nabla_grad(u_n)), v)*ufl.dx+ufl.inner(sigma(u_intermediario,p_n), epsilon(v))*ufl.dx+ufl.dot(p_n*n,v)*ufl.ds-ufl.dot(self.viscosidade*ufl.nabla_grad(u_n)*n,v)*ufl.ds
            A_u_ast=ufl.lhs(F)
            L_u_ast=ufl.rhs(F)

            #Passo 2
            A_p = ufl.dot(ufl.grad(p), ufl.grad(q)) * ufl.dx - q * ufl.dot(ufl.grad(p), n) * ufl.ds
            L_p = ufl.dot(ufl.grad(p_n), ufl.grad(q)) * ufl.dx - q * ufl.dot(ufl.grad(p_n), n) * ufl.ds - 1 / dt * q * ufl.div(u_ast) * ufl.dx
            #Passo 3
            A_u = 1 / dt * ufl.dot(u, v) * ufl.dx
            L_u = 1 / dt * ufl.dot(u_ast, v) * ufl.dx + ufl.dot(ufl.grad(p_n), v) * ufl.dx - ufl.dot(ufl.grad(p_novo), v) * ufl.dx

        else:
            raise NotImplementedError("Apenas escoamentos inviscidos sao aceitos no momento!")
        ##Passo 1
        bilinear1 = dfem.form(A_u_ast)
        linear1 = dfem.form(L_u_ast)
        A1 = dpetsc.assemble_matrix(bilinear1, bcs=condicoes_V)
        A1.assemble()
        b1 = dpetsc.create_vector(linear1)
        solver1 = dppetsc.KSP().create(self.malha.comm)
        solver1.setOperators(A1)
        solver1.setType(dppetsc.KSP.Type.PREONLY)
        solver1.getPC().setType(dppetsc.PC.Type.LU)
        ##Passo 2
        bilinear2 = dfem.form(A_p)
        A2 = dpetsc.assemble_matrix(bilinear2, bcs=condicoes_Q)
        A2.assemble()
        linear2 = dfem.form(L_p)
        b2 = dpetsc.create_vector(linear2)
        solver2 = dppetsc.KSP().create(self.malha.comm)
        solver2.setOperators(A2)
        solver2.setType(dppetsc.KSP.Type.PREONLY)
        solver2.getPC().setType(dppetsc.PC.Type.LU)
        ##Passo 3
        bilinear3 = dfem.form(A_u)
        A3 = dpetsc.assemble_matrix(bilinear3, bcs=condicoes_V)
        A3.assemble()
        linear3 = dfem.form(L_u)
        b3 = dpetsc.create_vector(linear3)
        solver3 = dppetsc.KSP().create(self.malha.comm)
        solver3.setOperators(A3)
        solver3.setType(dppetsc.KSP.Type.PREONLY)
        solver3.getPC().setType(dppetsc.PC.Type.LU)
        # Escrevendo os resultados em arquivo
        vtk_u = dio.VTKFile(self.malha.comm, "Saida/EF1/u.vtk", "w")
        vtk_p = dio.VTKFile(self.malha.comm, "Saida/EF1/p.vtk", "w")
        vtk_u.write_mesh(self.malha)
        vtk_u.write_function(u_n, 0)
        vtk_p.write_mesh(self.malha)
        vtk_p.write_function(p_n, 0)
        for t in tempos:
            print(t)
            ##Passo 1: encontrar a velocidade tentativa u_ast (ast=asterisco)
            ##Tentativa de reutilizar as matrizes de solucao
            # Update the right hand side reusing the initial vector
            with b1.localForm() as loc_b:
                loc_b.set(0)
            dpetsc.assemble_vector(b1, linear1)
            # Apply Dirichlet boundary condition to the vector
            dpetsc.apply_lifting(b1, [bilinear1], [condicoes_V])
            b1.ghostUpdate(addv=dppetsc.InsertMode.ADD_VALUES, mode=dppetsc.ScatterMode.REVERSE)
            dfem.set_bc(b1, condicoes_V)
            # Solve linear problem
            solver1.solve(b1, u_ast.vector)
            u_ast.x.scatter_forward()
            ##Passo 2: encontrar a pressao no passo n+1
            with b2.localForm() as loc_b:
                loc_b.set(0)
            dpetsc.assemble_vector(b2, linear2)
            dpetsc.apply_lifting(b2, [bilinear2], [condicoes_Q])
            b2.ghostUpdate(addv=dppetsc.InsertMode.ADD_VALUES, mode=dppetsc.ScatterMode.REVERSE)
            dfem.set_bc(b2, condicoes_Q)
            solver2.solve(b2, p_novo.vector)
            p_novo.x.scatter_forward()

            ##Passo 3: encontrar a velocidade no passo n+1
            with b3.localForm() as loc_b:
                loc_b.set(0)
            dpetsc.assemble_vector(b3, linear3)
            dpetsc.apply_lifting(b3, [bilinear3], [condicoes_V])
            b3.ghostUpdate(addv=dppetsc.InsertMode.ADD_VALUES, mode=dppetsc.ScatterMode.REVERSE)
            dfem.set_bc(b3, condicoes_V)
            solver3.solve(b3, u_novo.vector)
            u_novo.x.scatter_forward()

            u_n.interpolate(u_novo)
            p_n.interpolate(p_novo)
            ##Escrevendo os resultados em arquivo
            if np.isclose(t%0.5,0):
                vtk_u.write_function(u_n, t)
                vtk_p.write_function(p_n, t)

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
    problema = dpetsc.LinearProblem(a, L, bcs=[bc_entrada, bc_saida, bc_superior,
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
    import gmsh
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import tqdm.autonotebook

    from mpi4py import MPI
    from petsc4py import PETSc

    from dolfinx.cpp.mesh import to_type, cell_entity_type
    from dolfinx.fem import (Constant, Function, FunctionSpace,
                             assemble_scalar, dirichletbc, form, locate_dofs_topological, set_bc)
    from dolfinx.fem.petsc import (apply_lifting, assemble_matrix, assemble_vector,
                                   create_vector, create_matrix, set_bc)
    from dolfinx.graph import adjacencylist
    from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
    # from dolfinx.io import (VTXWriter, distribute_entity_data, gmshio)
    from dolfinx.io import distribute_entity_data, gmshio
    from dolfinx.mesh import create_mesh, meshtags_from_entities

    from ufl import (FacetNormal, FiniteElement, Identity, Measure, TestFunction, TrialFunction, VectorElement,
                     as_vector, div, dot, ds, dx, inner, lhs, grad, nabla_grad, rhs, sym)

    gmsh.initialize()

    L = 2.2
    H = 0.41
    c_x = c_y = 0.2
    r = 0.05
    gdim = 2
    mesh_comm = MPI.COMM_WORLD
    model_rank = 0
    if mesh_comm.rank == model_rank:
        rectangle = gmsh.model.occ.addRectangle(0, 0, 0, L, H, tag=1)
        obstacle = gmsh.model.occ.addDisk(c_x, c_y, 0, r, r)
    fluid_marker = 1
    if mesh_comm.rank == model_rank:
        volumes = gmsh.model.getEntities(dim=gdim)
        assert (len(volumes) == 1)
        gmsh.model.addPhysicalGroup(volumes[0][0], [volumes[0][1]], fluid_marker)
        gmsh.model.setPhysicalName(volumes[0][0], fluid_marker, "Fluid")
    inlet_marker, outlet_marker, wall_marker, obstacle_marker = 2, 3, 4, 5
    inflow, outflow, walls, obstacle = [], [], [], []
    if mesh_comm.rank == model_rank:
        boundaries = gmsh.model.getBoundary(volumes, oriented=False)
        for boundary in boundaries:
            center_of_mass = gmsh.model.occ.getCenterOfMass(boundary[0], boundary[1])
            if np.allclose(center_of_mass, [0, H / 2, 0]):
                inflow.append(boundary[1])
            elif np.allclose(center_of_mass, [L, H / 2, 0]):
                outflow.append(boundary[1])
            elif np.allclose(center_of_mass, [L / 2, H, 0]) or np.allclose(center_of_mass, [L / 2, 0, 0]):
                walls.append(boundary[1])
            else:
                obstacle.append(boundary[1])
        gmsh.model.addPhysicalGroup(1, walls, wall_marker)
        gmsh.model.setPhysicalName(1, wall_marker, "Walls")
        gmsh.model.addPhysicalGroup(1, inflow, inlet_marker)
        gmsh.model.setPhysicalName(1, inlet_marker, "Inlet")
        gmsh.model.addPhysicalGroup(1, outflow, outlet_marker)
        gmsh.model.setPhysicalName(1, outlet_marker, "Outlet")
        gmsh.model.addPhysicalGroup(1, obstacle, obstacle_marker)
        gmsh.model.setPhysicalName(1, obstacle_marker, "Obstacle")
    # Create distance field from obstacle.
    # Add threshold of mesh sizes based on the distance field
    # LcMax -                  /--------
    #                      /
    # LcMin -o---------/
    #        |         |       |
    #       Point    DistMin DistMax
    res_min = r / 3
    if mesh_comm.rank == model_rank:
        distance_field = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(distance_field, "EdgesList", obstacle)
        threshold_field = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(threshold_field, "IField", distance_field)
        gmsh.model.mesh.field.setNumber(threshold_field, "LcMin", res_min)
        gmsh.model.mesh.field.setNumber(threshold_field, "LcMax", 0.25 * H)
        gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", r)
        gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", 2 * H)
        min_field = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", [threshold_field])
        gmsh.model.mesh.field.setAsBackgroundMesh(min_field)
    if mesh_comm.rank == model_rank:
        gmsh.option.setNumber("Mesh.Algorithm", 8)
        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
        gmsh.model.mesh.generate(gdim)
        gmsh.model.mesh.setOrder(2)
        gmsh.model.mesh.optimize("Netgen")
    mesh, _, ft = gmshio.model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=gdim)
    ft.name = "Facet markers"
    t = 0
    T = 8  # Final time
    dt = 1 / 1600  # Time step size
    num_steps = int(T / dt)
    k = Constant(mesh, PETSc.ScalarType(dt))
    mu = Constant(mesh, PETSc.ScalarType(0.001))  # Dynamic viscosity
    rho = Constant(mesh, PETSc.ScalarType(1))  # Density
    v_cg2 = VectorElement("Lagrange", mesh.ufl_cell(), 2)
    s_cg1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    V = FunctionSpace(mesh, v_cg2)
    Q = FunctionSpace(mesh, s_cg1)

    fdim = mesh.topology.dim - 1

    # Define boundary conditions

    class InletVelocity():
        def __init__(self, t):
            self.t = t

        def __call__(self, x):
            values = np.zeros((gdim, x.shape[1]), dtype=PETSc.ScalarType)
            values[0] = 4 * 1.5 * np.sin(self.t * np.pi / 8) * x[1] * (0.41 - x[1]) / (0.41 ** 2)
            return values

    # Inlet
    u_inlet = Function(V)
    inlet_velocity = InletVelocity(t)
    u_inlet.interpolate(inlet_velocity)
    bcu_inflow = dirichletbc(u_inlet, locate_dofs_topological(V, fdim, ft.find(inlet_marker)))
    # Walls
    u_nonslip = np.array((0,) * mesh.geometry.dim, dtype=PETSc.ScalarType)
    bcu_walls = dirichletbc(u_nonslip, locate_dofs_topological(V, fdim, ft.find(wall_marker)), V)
    # Obstacle
    bcu_obstacle = dirichletbc(u_nonslip, locate_dofs_topological(V, fdim, ft.find(obstacle_marker)), V)
    bcu = [bcu_inflow, bcu_obstacle, bcu_walls]
    # Outlet
    bcp_outlet = dirichletbc(PETSc.ScalarType(0), locate_dofs_topological(Q, fdim, ft.find(outlet_marker)), Q)
    bcp = [bcp_outlet]
    u = TrialFunction(V)
    v = TestFunction(V)
    u_ = Function(V)
    u_.name = "u"
    u_s = Function(V)
    u_n = Function(V)
    u_n1 = Function(V)
    p = TrialFunction(Q)
    q = TestFunction(Q)
    p_ = Function(Q)
    p_.name = "p"
    phi = Function(Q)
    f = Constant(mesh, PETSc.ScalarType((0, 0)))
    F1 = rho / k * dot(u - u_n, v) * dx
    F1 += inner(dot(1.5 * u_n - 0.5 * u_n1, 0.5 * nabla_grad(u + u_n)), v) * dx
    F1 += 0.5 * mu * inner(grad(u + u_n), grad(v)) * dx - dot(p_, div(v)) * dx
    F1 += dot(f, v) * dx
    a1 = form(lhs(F1))
    L1 = form(rhs(F1))
    A1 = create_matrix(a1)
    b1 = create_vector(L1)
    a2 = form(dot(grad(p), grad(q)) * dx)
    L2 = form(-rho / k * dot(div(u_s), q) * dx)
    A2 = assemble_matrix(a2, bcs=bcp)
    A2.assemble()
    b2 = create_vector(L2)
    a3 = form(rho * dot(u, v) * dx)
    L3 = form(rho * dot(u_s, v) * dx - k * dot(nabla_grad(phi), v) * dx)
    A3 = assemble_matrix(a3)
    A3.assemble()
    b3 = create_vector(L3)
    # Solver for step 1
    solver1 = PETSc.KSP().create(mesh.comm)
    solver1.setOperators(A1)
    solver1.setType(PETSc.KSP.Type.BCGS)
    pc1 = solver1.getPC()
    pc1.setType(PETSc.PC.Type.JACOBI)

    # Solver for step 2
    solver2 = PETSc.KSP().create(mesh.comm)
    solver2.setOperators(A2)
    solver2.setType(PETSc.KSP.Type.MINRES)
    pc2 = solver2.getPC()
    pc2.setType(PETSc.PC.Type.HYPRE)
    pc2.setHYPREType("boomeramg")

    # Solver for step 3
    solver3 = PETSc.KSP().create(mesh.comm)
    solver3.setOperators(A3)
    solver3.setType(PETSc.KSP.Type.CG)
    pc3 = solver3.getPC()
    pc3.setType(PETSc.PC.Type.SOR)
    n = -FacetNormal(mesh)  # Normal pointing out of obstacle
    dObs = Measure("ds", domain=mesh, subdomain_data=ft, subdomain_id=obstacle_marker)
    u_t = inner(as_vector((n[1], -n[0])), u_)
    drag = form(2 / 0.1 * (mu / rho * inner(grad(u_t), n) * n[1] - p_ * n[0]) * dObs)
    lift = form(-2 / 0.1 * (mu / rho * inner(grad(u_t), n) * n[0] + p_ * n[1]) * dObs)
    if mesh.comm.rank == 0:
        C_D = np.zeros(num_steps, dtype=PETSc.ScalarType)
        C_L = np.zeros(num_steps, dtype=PETSc.ScalarType)
        t_u = np.zeros(num_steps, dtype=np.float64)
        t_p = np.zeros(num_steps, dtype=np.float64)
    tree = bb_tree(mesh, mesh.geometry.dim)
    points = np.array([[0.15, 0.2, 0], [0.25, 0.2, 0]])
    cell_candidates = compute_collisions_points(tree, points)
    colliding_cells = compute_colliding_cells(mesh, cell_candidates, points)
    front_cells = colliding_cells.links(0)
    back_cells = colliding_cells.links(1)
    if mesh.comm.rank == 0:
        p_diff = np.zeros(num_steps, dtype=PETSc.ScalarType)
    from pathlib import Path
    folder = Path("results")
    folder.mkdir(exist_ok=True, parents=True)
    # vtx_u = VTXWriter(mesh.comm, "dfg2D-3-u.bp", [u_], engine="BP4")
    # vtx_p = VTXWriter(mesh.comm, "dfg2D-3-p.bp", [p_], engine="BP4")
    # vtx_u.write(t)
    # vtx_p.write(t)
    progress = tqdm.autonotebook.tqdm(desc="Solving PDE", total=num_steps)
    for i in range(num_steps):
        progress.update(1)
        # Update current time step
        t += dt
        # Update inlet velocity
        inlet_velocity.t = t
        u_inlet.interpolate(inlet_velocity)

        # Step 1: Tentative velocity step
        A1.zeroEntries()
        assemble_matrix(A1, a1, bcs=bcu)
        A1.assemble()
        with b1.localForm() as loc:
            loc.set(0)
        assemble_vector(b1, L1)
        apply_lifting(b1, [a1], [bcu])
        b1.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b1, bcu)
        solver1.solve(b1, u_s.vector)
        u_s.x.scatter_forward()

        # Step 2: Pressure corrrection step
        with b2.localForm() as loc:
            loc.set(0)
        assemble_vector(b2, L2)
        apply_lifting(b2, [a2], [bcp])
        b2.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b2, bcp)
        solver2.solve(b2, phi.vector)
        phi.x.scatter_forward()

        p_.vector.axpy(1, phi.vector)
        p_.x.scatter_forward()

        # Step 3: Velocity correction step
        with b3.localForm() as loc:
            loc.set(0)
        assemble_vector(b3, L3)
        b3.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        solver3.solve(b3, u_.vector)
        u_.x.scatter_forward()

        # Write solutions to file
        # vtx_u.write(t)
        # vtx_p.write(t)

        # Update variable with solution form this time step
        with u_.vector.localForm() as loc_, u_n.vector.localForm() as loc_n, u_n1.vector.localForm() as loc_n1:
            loc_n.copy(loc_n1)
            loc_.copy(loc_n)

        # Compute physical quantities
        # For this to work in paralell, we gather contributions from all processors
        # to processor zero and sum the contributions.
        drag_coeff = mesh.comm.gather(assemble_scalar(drag), root=0)
        lift_coeff = mesh.comm.gather(assemble_scalar(lift), root=0)
        p_front = None
        if len(front_cells) > 0:
            p_front = p_.eval(points[0], front_cells[:1])
        p_front = mesh.comm.gather(p_front, root=0)
        p_back = None
        if len(back_cells) > 0:
            p_back = p_.eval(points[1], back_cells[:1])
        p_back = mesh.comm.gather(p_back, root=0)
        if mesh.comm.rank == 0:
            t_u[i] = t
            t_p[i] = t - dt / 2
            C_D[i] = sum(drag_coeff)
            C_L[i] = sum(lift_coeff)
            # Choose first pressure that is found from the different processors
            for pressure in p_front:
                if pressure is not None:
                    p_diff[i] = pressure[0]
                    break
            for pressure in p_back:
                if pressure is not None:
                    p_diff[i] -= pressure[0]
                    break
    # vtx_u.close()
    # vtx_p.close()


if __name__ == "__main__" :
    import AerofolioFino

    aerofolio = AerofolioFino.AerofolioFinoNACA4([0.04, 0.4, 0.12], 0, 1)
    # nome_malha=malha_aerofolio(aerofolio, nome_modelo="4412 grosseiro", n_pontos_contorno=100)
    nome_malha = 'Malha/4412 grosseiro.msh'
    exemplo()
    solucao = SolucaoEscoamento(aerofolio, nome_malha, n_pontos_contorno=1000, gerar_malha=False, caso="viscoso")
    # solucao.ordena_contorno()
    # print(solucao.calcula_forcas())
    print("?")
