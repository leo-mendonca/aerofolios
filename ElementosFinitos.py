from mpi4py import MPI
from dolfinx import mesh as dmesh
import dolfinx.io as dio
import numpy as np
import os
from petsc4py.PETSc import ScalarType
import ufl  # Unified Form Language. Linguagem para definicao de problemas de elementos finitos e forma fraca
import gmsh

geo = gmsh.model.geo  # definindo um alias para o modulo de geometria do gmsh
import dolfinx.io.gmshio

# MPI.COMM_WORLD permite a paralelizacao de uma mesma malha entre processadores diferentes
from dolfinx import fem as dfem


def malha_aerofolio(aerofolio, nome_modelo="modelo") :
    '''Gera uma malha no gmsh correspondendo a regiao em torno do aerofolio'''
    ##TODO implementar
    contornos = {"entrada" : 1, "saida" : 2, "superior" : 3, "inferior" : 4, "af_superior" : 5, "af_inferior" : 6}
    tag_fis={} #tags dos grupos fisicos
    af_tamanho = 0.01
    tamanho=0.1
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

    # teste com circulo em vez de aerofolio
    circ_E = geo.add_point(1, 0, 0, af_tamanho)
    circ_N = geo.add_point(1 / 2, 1 / 2, 0, af_tamanho)
    circ_O = geo.add_point(0, 0, 0, af_tamanho)
    circ_S = geo.add_point(1 / 2, -1 / 2, 0, af_tamanho)
    circ_centro = geo.add_point(1 / 2, 0, 0, af_tamanho)
    af_sup = geo.add_circle_arc(circ_E, circ_centro, circ_O, tag=contornos["af_superior"])
    af_inf = geo.add_circle_arc(circ_O, circ_centro, circ_E, tag=contornos["af_inferior"])



    ###Definindo as superficies para simulacao
    geo.add_curve_loop([-af_sup, -af_inf], tag=2)  # superficie do aerofolio
    geo.add_curve_loop([-1, 4, 2, -3], tag=1)  # superficie externa
    geo.add_plane_surface([1, 2], tag=1)  # superficie do escoamento

    ##Criando grupos fisicos correspondendo a cada elemento da simulacao
    tag_fis["af"]=geo.add_physical_group(1, [af_sup, af_inf])
    tag_fis["entrada"]=geo.add_physical_group(1, [contornos["entrada"]])
    tag_fis["saida"]=geo.add_physical_group(1, [contornos["saida"]])
    tag_fis["superior"]=geo.add_physical_group(1, [contornos["superior"]])
    tag_fis["inferior"]=geo.add_physical_group(1, [contornos["inferior"]])
    tag_fis["escoamento"]=geo.add_physical_group(2, [1])

    ###Sincronizar as modificacoes geometricas e gerar a malha
    geo.synchronize()  # necessario!
    gmsh.model.mesh.generate(2)  # gera a malha

    gmsh.write(os.path.join("Malha", f"{nome_modelo}.msh"))  # salva o arquivo da malha

    ###Exportando a malha para o dolfinx
    malha, cell_tags, facet_tags = dio.gmshio.read_from_msh(os.path.join("Malha", f"{nome_modelo}.msh"), MPI.COMM_WORLD,  rank=0, gdim=2)
    V = dfem.FunctionSpace(malha, ("CG", 2))  # define o espaco de funcao teste e tentativa
    regioes_fisicas = dfem.locate_dofs_topological(V, entity_dim=1, entities="???") #TODO completar e ver se funciona


    ##Encerrando o gmsh
    gmsh.finalize()


def calculo_aerofolio(aerofolio) :
    '''
    Calcula as propriedades aerodinamicas de um aerofolio
    :param aerofolio: objeto da classe AerofolioFino
    '''
    ##TODO implementar
    y_1, y_2 = -1, 1
    x_1, x_2 = -2, 3
    U0 = aerofolio.U0
    alfa = aerofolio.alfa
    n_pontos_aerofolio = 100
    domain = dmesh.create_box_mesh(MPI.COMM_WORLD, [np.array([x_1, y_1, 0.]), np.array([x_2, y_2, 0.])], [1000, 1000, 1], dmesh.CellType.triangle)
    V = fem.FunctionSpace(domain, ("CG", 2))  ##CG eh a familia de polinomios interpoladores de Lagrange, 2 eh a ordem do polinomio
    psi_entrada = lambda x : x[1] * U0  # define-se psi=0 em y=0
    u_in = fem.Function(V)
    u_out = fem.Function(V)
    u_in.interpolate(psi_entrada)
    u_out.interpolate(psi_entrada)


def exemplo() :
    # from dolfinx.fem import FunctionSpace
    ##Vamos resolver a equacao de Poisson: \del² u = f
    ##O problema foi manufaturado com f=-6 para ter solucao exata u = 1+x²+2y²
    domain = dmesh.create_unit_square(MPI.COMM_WORLD, 8, 8, dmesh.CellType.quadrilateral)
    V = fem.FunctionSpace(domain, ("CG", 2))  ##CG eh a familia de polinomios interpoladores de Lagrange, 2 eh a ordem do polinomio
    uD = fem.Function(V)
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
    malha_aerofolio(None)
