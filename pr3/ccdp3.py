#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Robótica Computacional - 
# Grado en Ingeniería Informática (Cuarto)
# Práctica: Resolución de la cinemática inversa mediante CCD
#           (Cyclic Coordinate Descent).

import sys
from math import *
import numpy as np
import matplotlib.pyplot as plt
import colorsys as cs

# ******************************************************************************
# Declaración de funciones

def muestra_origenes(O,final=0):
  # Muestra los orígenes de coordenadas para cada articulación
  print('Origenes de coordenadas:')
  for i in range(len(O)):
    print('(O'+str(i)+')0\t= '+str([round(j,3) for j in O[i]]))
  if final:
    print('E.Final = '+str([round(j,3) for j in final]))

def muestra_robot(O,obj):
  # Muestra el robot graficamente
  plt.figure()
  plt.xlim(-L,L)
  plt.ylim(-L,L)
  T = [np.array(o).T.tolist() for o in O]
  for i in range(len(T)):
    plt.plot(T[i][0], T[i][1], '-o', color=cs.hsv_to_rgb(i/float(len(T)),1,1))
  plt.plot(obj[0], obj[1], '*')
  plt.pause(0.0001)
  plt.show()
  
#  input()
  plt.close()

def matriz_T(d,th,a,al):
   # los angulos son en rad
  return [[cos(th), -sin(th)*cos(al),  sin(th)*sin(al), a*cos(th)]
         ,[sin(th),  cos(th)*cos(al), -sin(al)*cos(th), a*sin(th)]
         ,[      0,          sin(al),          cos(al),         d]
         ,[      0,                0,                0,         1]
         ]

def cin_dir(th,a):
  #Sea 'th' el vector de thetas
  #Sea 'a'  el vector de longitudes
  # Devuelve lista de coordenadas x e y de una de los ...
  # o =[[x00.y00], [x10,y10], [x20,y20]]
  T = np.identity(4)
  o = [[0,0]]
  for i in range(len(th)):
    T = np.dot(T,matriz_T(0,th[i],a[i],0))
    tmp=np.dot(T,[0,0,0,1])
    o.append([tmp[0],tmp[1]])
  return o

# ******************************************************************************
# Cálculo de la cinemática inversa de forma iterativa por el método CCD

# valores articulares arbitrarios para la cinemática directa inicial
th=[0.,0.,0.]
a =[5.,0.,5.]

prismaticos = [False, True, False]
# Con psrimasticos
thmin = np.array([-pi/2, 0.0, -pi/2])
thmax = np.array([ pi/2, 0.0,  pi/2])


#sin primasticos
#thmin = np.array([-pi/2, -pi/2, -pi/2])
#thmax = np.array([ pi/2,  pi/2,  pi/2])

prismaticas = []
L = sum(a) # variable para representación gráfica
EPSILON = .01

#plt.ion() # modo interactivo

# introducción del punto para la cinemática inversa
if len(sys.argv) != 3:
  sys.exit("python " + sys.argv[0] + " x y")
objetivo=[float(i) for i in sys.argv[1:]]
O=cin_dir(th,a)
#O=zeros(len(th)+1) # Reservamos estructura en memoria
 # Calculamos la posicion inicial
print ("- Posicion inicial:")
muestra_origenes(O)


numart = len(th) # Numero de articulaciones del manipulador
dist = float("inf")
prev = 0.
iteracion = 1
while (dist > EPSILON and abs(prev-dist) > EPSILON/100.):
  prev = dist
  O=[cin_dir(th,a)] #Es una lista de o, es decir, una lista de posiciones de cinematica directa
  # Para cada combinación de articulaciones:
  # o =[[x00.y00], [x10,y10], [x20,y20]] iteracion inical
  # o =[[x00.y00], [x10,y10], [x20,y20]] una correcion...
  # etc
  for i in range(numart):
    artactual = numart - i - 1 # ver lo del - 1
    if not prismaticas(artactual):
      # SI es prismatico
      o = O[0]                          # configuración actual (lista de [x,y])
      pj = np.array(o[artactual])       # posición de la articulación actual
      pt = np.array(objetivo)           # objetivo

      # omega = orientación del eje x de la articulación (suma de las rotaciones anteriores)
      omega = sum(th[:artactual+1])
      eje = np.array([cos(omega), sin(omega)])  # vector unitario del eje prismático

      # proyección del vector (pt - pj) sobre el eje -> longitud deseada desde pj
      longitud_objetivo = np.dot(pt - pj, eje)

      # limitar según los límites (se reutilizan thmin/thmax como qmin/qmax para prismáticos)
      longitud_objetivo = max(min(longitud_objetivo, thmax[artactual]), thmin[artactual])

      # aplicar corrección a la longitud (a[artactual])
      a[artactual] = longitud_objetivo

      # actualizar configuración directa tras el cambio prismático
      O.append(cin_dir(th, a))

    else:
      # Hay que hacer un if si es prismatico o no y depende de si lo es o no ahcer una cosa o no
      # cálculo de la cinemática inversa:
      # obtener los alfas, las correcciones de theta, aplicarlas
      o = O[0]                          # configuración actual (lista de [x,y])
      pj = np.array(o[artactual])       # posición de la articulación actual
      pe = np.array(o[-1])              # posición del efector final actual
      pt = np.array(objetivo)           # objetivo
      v1 = pe - pj
      v2 = pt - pj
      n1 = np.linalg.norm(v1)
      n2 = np.linalg.norm(v2)
      if n1 < 1e-8 or n2 < 1e-8:
        deltathita = 0.0
      else:
        # ángulo de corrección: atan2(det, dot)
        det = v1[0]*v2[1] - v1[1]*v2[0]
        dot = v1[0]*v2[0] + v1[1]*v2[1]
        deltathita = atan2(det, dot)
      th[artactual] = th[artactual] + deltathita
      # Pasar th[artactual] al rango (-pi, pi)
      th[artactual] = (th[artactual] + pi) % (2*pi) - pi
      
      
      O.append(cin_dir(th,a))

  dist = np.linalg.norm(np.subtract(objetivo,O[-1][-1]))
  print ("\n- Iteracion " + str(iteracion) + ':')
  muestra_origenes(O[-1])
  muestra_robot(O,objetivo)
  print ("Distancia al objetivo = " + str(round(dist,5)))
  iteracion+=1
  O[0]=O[-1]

if dist <= EPSILON:
  print ("\n" + str(iteracion) + " iteraciones para converger.")
else:
  print ("\nNo hay convergencia tras " + str(iteracion) + " iteraciones.")
print ("- Umbral de convergencia epsilon: " + str(EPSILON))
print ("- Distancia al objetivo:          " + str(round(dist,5)))
print ("- Valores finales de las articulaciones:")
for i in range(len(th)):
  print ("  theta" + str(i+1) + " = " + str(round(th[i],3)))
for i in range(len(th)):
  print ("  L" + str(i+1) + "     = " + str(round(a[i],3)))
