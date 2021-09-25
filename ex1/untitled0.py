#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 21:48:35 2020

@author: fabian
"""
import netgen.gui
from ngsolve import *
from netgen.geom2d import unit_square
from netgen.csg import unit_cube
mesh = Mesh(unit_square.GenerateMesh(maxh=0.3))
order=3
fes = H1(mesh, order=order)
gfu = GridFunction(fes)
Draw(gfu, sd=4)


mesh = Mesh(unit_square.GenerateMesh(maxh=0.3))
# basis functions on edge nr:
edge_dofs = fes.GetDofNrs(NodeId(EDGE,10))
print("edge_dofs =", edge_dofs)
SetVisualization(min=-0.05, max=0.05)
gfu.vec[:] = 0
gfu.vec[edge_dofs[0]] = 1
Redraw()