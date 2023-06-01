# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# ### Understanding the poisson solver

# I want to understand how the Poisson equation $\nabla^2 \phi = \frac{S(\vec{x} )}{k}$ is solved in underworld3.

import underworld3 as uw

## create our mesh
mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize= 1.0 / (24), qdegree=5)

## define our field on the mesh
phi = uw.discretisation.MeshVariable(r"\phi", mesh, 1, degree=2)

# I'll try use the Poisson solver to solve the Poisson equation

poisson_solver = uw.systems.Poisson(mesh, u_Field=phi)

# As far as I can tell, the poisson solver uses the flux $\vec{F} = k \nabla \phi$. Once we have this flux, uw3 in the background can write an equation of the form $\nabla \cdot \vec{F} - f_0 = 0$. If we have $f_0 = S( \vec{x})$, then this is the equation $\nabla \cdot (k \nabla \phi) = S (
# \vec{x})$. If $k$ is constant in space, then this is the original equation. uw3 can then go away and convert this into a weak form and then solve using finite element methods.

# To define the flux $F$, we use a consitutive diffusion model and set the diffusivity in this model to $k$

poisson_solver.constitutive_model = uw.systems.constitutive_models.DiffusionModel(mesh.dim)
poisson_solver.constitutive_model.Parameters.diffusivity = 1 ## we will make it a constant for now

# To add the source term $S(\vec{x})$, we change the poisson_solver.f property. For simplicity here, lets set $S(\vec{x}) = 0$. 

poisson_solver.f = 0 

# Then, we can give the solver some boundary conditions, namely $\phi(x,0) = 1$ and $\phi(x, 1) = 0$.

poisson_solver.add_dirichlet_bc(1, "Bottom")
poisson_solver.add_dirichlet_bc(0, "Top")

# We can now solve the poisson equation $\nabla^2 \phi = 0$

poisson_solver.solve()

# Store our solution to the problem

with mesh.access():
    mesh_numerical_soln = uw.function.evaluate(poisson_solver.u.fn, mesh.data)

# Now, lets print everything out so that we can see our result

from mpi4py import MPI ## library for displaying everything

if MPI.COMM_WORLD.size == 1:

    import numpy as np
    import pyvista as pv

    pv.global_theme.background = "white"
    pv.global_theme.window_size = [500, 500]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = "panel"
    pv.global_theme.smooth_shading = True

    mesh.vtk("ignore_mesh.vtk")
    pvmesh = pv.read("ignore_mesh.vtk")

    pvmesh.point_data["phi"] = mesh_numerical_soln

    sargs = dict(interactive=True)  # doesn't appear to work :(
    pl = pv.Plotter()

    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Black",
        show_edges=True,
        scalars="phi",
        use_transparency=False,
        opacity=0.5,
        scalar_bar_args=sargs,
    )

    pl.camera_position = "xy"

    pl.show(cpos="xy")
    pl.screenshot(filename="test.png")

# And thats our solution to Laplaces equation $\nabla^2 \phi = 0$ on $[0,1] \times [0,1]$ with $\phi(x,0) = 1, \phi(x,1)= 0$

# Now, lets understand what happens if we change the variable $k$. If we set $k=0$, then the flux term term $\vec{F} = k \nabla \phi = 0$. Then, the whole solution does not evolve as there is no flux term. Thus, we get no change from the initial conditions.

# Lets plot that

# Set $k=0$:

poisson_solver.constitutive_model.Parameters.diffusivity = 0

# Keep our bc and source function

poisson_solver.add_dirichlet_bc(1, "Bottom")
poisson_solver.add_dirichlet_bc(0, "Top")
poisson_solver.f = 0

# Solve our poisson_solver

poisson_solver.solve()

# +
with mesh.access():
    mesh_numerical_soln = uw.function.evaluate(poisson_solver.u.fn, mesh.data)
    
if MPI.COMM_WORLD.size == 1:

    import numpy as np
    import pyvista as pv

    pv.global_theme.background = "white"
    pv.global_theme.window_size = [500, 500]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = "panel"
    pv.global_theme.smooth_shading = True

    mesh.vtk("ignore_mesh.vtk")
    pvmesh = pv.read("ignore_mesh.vtk")

    pvmesh.point_data["phi"] = mesh_numerical_soln

    sargs = dict(interactive=True)  # doesn't appear to work :(
    pl = pv.Plotter()

    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Black",
        show_edges=True,
        scalars="phi",
        use_transparency=False,
        opacity=0.5,
        scalar_bar_args=sargs,
    )

    pl.camera_position = "xy"

    pl.show(cpos="xy")
    pl.screenshot(filename="test.png")
    
# -

# We can see that there is no propigation of the boundary, even though we are solving the same equation.

# Now lets solve a more complicated problem with a source term, $\nabla^2 \phi = \sin(2 \pi x) \cos(2 \pi y)$. We will do this on the same domain with boundary conditions $\phi(0,y)=\phi(1,y)=0$ and $\phi(x,0) = \phi(x,1)= - \frac{\sin(2 \pi  x)}{8 \pi^2}$. To do this, we set $k=1$ and $S(\vec{x}) = \sin(2 \pi x) \cos(2 \pi y)$

poisson_solver.constitutive_model.Parameters.diffusivity = 1

# To set the source term $S(\vec{x})$ we must make poisson_solver.f $\sin(2 \pi x) \cos(2 \pi y)$. To make it a function of the cooridnates we get the mesh coodinates using mesh.X. We pass the function poisson_solve.f in symbolically using a sympy expression.

# +
import math
import sympy

x,y = mesh.X ## access the expressions for the coordinates of the mesh
source_term = sympy.sin( 2* sympy.pi * x) * sympy.cos(2* sympy.pi * y) ## write the expression for sin(x) cos(y) using sympy
# -

# You can see the source term is a symbolic expression

source_term

poisson_solver.f = -source_term

poisson_solver._f0

poisson_solver._f1

# We give the boundary conditions on the edges of the domain

# +
top_boundary = - sympy.sin(2 * sympy.pi * x)/(8 * sympy.pi**2)
bottom_boundary = - sympy.sin(2 * sympy.pi * x)/(8 * sympy.pi**2)

poisson_solver.add_dirichlet_bc(0, "Left")
poisson_solver.add_dirichlet_bc(0, "Right") 
poisson_solver.add_dirichlet_bc(top_boundary, "Top")
poisson_solver.add_dirichlet_bc(bottom_boundary, "Bottom")
# -

# Now we solve the system

poisson_solver.solve()

# +
with mesh.access():
    mesh_numerical_soln = uw.function.evaluate(poisson_solver.u.fn, mesh.data)
    
if MPI.COMM_WORLD.size == 1:

    import numpy as np
    import pyvista as pv

    pv.global_theme.background = "white"
    pv.global_theme.window_size = [500, 500]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = "panel"
    pv.global_theme.smooth_shading = True

    mesh.vtk("ignore_mesh.vtk")
    pvmesh = pv.read("ignore_mesh.vtk")

    pvmesh.point_data["phi"] = mesh_numerical_soln

    sargs = dict(interactive=True)  # doesn't appear to work :(
    pl = pv.Plotter()

    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Black",
        show_edges=True,
        scalars="phi",
        use_transparency=False,
        opacity=0.5,
        scalar_bar_args=sargs,
    )

    pl.camera_position = "xy"

    pl.show(cpos="xy")
    pl.screenshot(filename="numerical.png")
# -

# The Poisson equation $\nabla^2 \phi = \sin(2 \pi x) \cos( 2 \pi y)$ with these boundary conditions has the analytic solution $\phi = -\frac{\cos(2 \pi y) \sin(2 \pi x)}{8 \pi^2}$. Lets now plot that and see how it compares to our solution

analytic_fn = -sympy.cos(2 * sympy.pi * y) * sympy.sin(2 * sympy.pi * x)/(8 * sympy.pi**2)
analytic_fn

# +

with mesh.access():
    mesh_analytic_soln = uw.function.evaluate(analytic_fn, mesh.data)
# -

if MPI.COMM_WORLD.size == 1:

    import numpy as np
    import pyvista as pv

    pv.global_theme.background = "white"
    pv.global_theme.window_size = [500, 500]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = "panel"
    pv.global_theme.smooth_shading = True

    mesh.vtk("ignore_mesh.vtk")
    pvmesh = pv.read("ignore_mesh.vtk")

    pvmesh.point_data["phiAnalytic"] = mesh_analytic_soln

    sargs = dict(interactive=True)  # doesn't appear to work :(
    pl = pv.Plotter()

    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Black",
        show_edges=True,
        scalars="phiAnalytic",
        use_transparency=False,
        opacity=0.5,
        scalar_bar_args=sargs,
    )

    pl.camera_position = "xy"

    pl.show(cpos="xy")
    pl.screenshot(filename="analytic.png")

# Lets looks at the difference here, why is this happening?
#

if MPI.COMM_WORLD.size == 1:

    import numpy as np
    import pyvista as pv

    pv.global_theme.background = "white"
    pv.global_theme.window_size = [500, 500]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = "panel"
    pv.global_theme.smooth_shading = True

    mesh.vtk("ignore_mesh.vtk")
    pvmesh = pv.read("ignore_mesh.vtk")

    pvmesh.point_data["diff"] = mesh_analytic_soln - mesh_numerical_soln

    sargs = dict(interactive=True)  # doesn't appear to work :(
    pl = pv.Plotter()

    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Black",
        show_edges=True,
        scalars="diff",
        use_transparency=False,
        opacity=0.5,
        scalar_bar_args=sargs,
    )

    pl.camera_position = "xy"

    pl.show(cpos="xy")
    pl.screenshot(filename="difference.png")

# If we look at the analytic solution (analytic.png) and the numerical solution (numerical.png) they are quite different at the top and bottom boundaries. This is clear when looking at the difference between the numerical and analytic solutions (difference.png). Through a little bit of testing, this difference doesnt seem to be a function of mesh refinement, mesh degree or meshVariable degree.

poisson_solver


