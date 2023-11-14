# from mpi4py import MPI
# from dolfinx import log as dlog

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

viscosidade=1 #viscosidade dinamica do fluido

def exporta_valores(u, t, malha, path):
    '''Exporta os valores de u para um arquivo .csv'''
    #TODO fazer

    return

def calculo_aerofolio(aerofolio) :
    '''
    Calcula as propriedades aerodinamicas de um aerofolio
    :param aerofolio: objeto da classe AerofolioFino
    '''
    ##TODO implementar

    nome_arquivo = Malha.malha_aerofolio(aerofolio, aerofolio.nome)



class SolucaoEscoamento2 :
    def __init__(self, aerofolio, nome_malha, viscosidade=1, n_pontos_contorno=Malha.n_pontos_contorno_padrao, gerar_malha=True, caso="inviscido") :
        self.aerofolio=aerofolio
        self.n_pontos_contorno=n_pontos_contorno
        self.viscosidade=viscosidade #viscosidade cinematica do fluido
        if gerar_malha :
            nome_malha= Malha.malha_aerofolio(aerofolio, nome_malha, n_pontos_contorno)

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

class FEA(object):
    '''Classe para resolucao do escoamento usando o metodo de elementos finitos de Galerkin manualmente'''
    def __init__(self, nome_malha, tag_fis, aerofolio, velocidade=None):
        '''
        :param nome_malha: Nome do arquivo .msh produzido pelo gmsh. Deve ser uma malha de ordem 2
        :param tag_fis: dicionario contendo a tag de cada grupo fisico da malha
        :param aerofolio: AerofolioFino dado como entrada na simulacao
        :param velocidade: velocidade do escoamento, caso seja diferente daquela prevista pelo aerofolio
        '''
        if velocidade is None:
            velocidade=aerofolio.U0
        self.velocidade=velocidade
        self.aerofolio=aerofolio
        self.nos, self.x_nos, self.elementos, self.nos_cont, self.x_cont = Malha.ler_malha(nome_malha, tag_fis)
        self.nos_o1, self.elementos_o1=Malha.reduz_ordem(self.elementos)









if __name__ == "__main__" :
    import AerofolioFino

    aerofolio = AerofolioFino.AerofolioFinoNACA4([0.04, 0.4, 0.12], 0, 1)
    # nome_malha=malha_aerofolio(aerofolio, nome_modelo="4412 grosseiro", n_pontos_contorno=100)
    nome_malha = 'Malha/4412 grosseiro.msh'
    solucao = SolucaoEscoamento2(aerofolio, nome_malha, n_pontos_contorno=1000, gerar_malha=False, caso="viscoso")
    # solucao.ordena_contorno()
    # print(solucao.calcula_forcas())
    print("?")
