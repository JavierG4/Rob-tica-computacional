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

def normalize(rads):
  """Normalize angle to the range (-pi, pi]."""
  while rads > pi:
    rads -= 2 * pi
  while rads <= -pi:
    rads += 2 * pi
  return rads

def muestra_origenes(O, final=0):
  # Muestra los orígenes de coordenadas para cada articulación
  print('Origenes de coordenadas:')
  for i in range(len(O)):
    print('(O'+str(i)+')0\t= '+str([round(j,3) for j in O[i]]))
  if final:
    print('E.Final = '+str([round(j,3) for j in final]))

def muestra_robot(O, obj, anim=False):
  # Muestra el robot graficamente
  if anim:
    plt.clf()
    plt.xlim(-L,L)
    plt.ylim(-L,L)
    plt.grid(True)
  else:
    plt.figure()
    plt.xlim(-L,L)
    plt.ylim(-L,L)
    plt.grid(True)
    
  T = [np.array(o).T.tolist() for o in O]
  for i in range(len(T)):
    plt.plot(T[i][0], T[i][1], '-o', color=cs.hsv_to_rgb(i/float(len(T)),1,1))
  plt.plot(obj[0], obj[1], '*')
  plt.title("Iteración CCD")
  plt.xlabel("X")
  plt.ylabel("Y")
  
  if anim:
    plt.pause(0.3)
  else:
    plt.show()
    plt.close()

def matriz_T(d, th, a, al):
   
  return [[cos(th), -sin(th)*cos(al),  sin(th)*sin(al), a*cos(th)]
         ,[sin(th),  cos(th)*cos(al), -sin(al)*cos(th), a*sin(th)]
         ,[      0,          sin(al),          cos(al),         d]
         ,[      0,                0,                0,         1]
         ]

def cin_dir(th, a):
  #Sea 'th' el vector de thetas
  #Sea 'a'  el vector de longitudes
  T = np.identity(4)
  o = [[0,0]]
  for i in range(len(th)):
    T = np.dot(T, matriz_T(0, th[i], a[i], 0))
    tmp = np.dot(T,[0, 0, 0, 1])
    o.append([tmp[0], tmp[1]])
  return o

# ******************************************************************************
# Cálculo de la cinemática inversa de forma iterativa por el método CCD

# valores articulares arbitrarios para la cinemática directa inicial
th           = [0.,     0.,     0., 0.]
a            = [5.,     5.,     5., 5.]
joint_type   = [0,       1,      0, 0] # 0 = rotatoria, 1 = prismática
upper_limits = [180, 10, 180, 180]
lower_limits = [-179, 1, -180, -180]

for i in range(len(upper_limits)):
  if (joint_type[i] == 0):
    upper_limits[i] = (upper_limits[i] * pi) / 180
    upper_limits[i] = normalize(upper_limits[i])
for i in range(len(lower_limits)):
  if (joint_type[i] == 0):
    lower_limits[i] = (lower_limits[i] * pi) / 180
    lower_limits[i] = normalize(lower_limits[i])
  
L = sum(a) # variable para representación gráfica
EPSILON = .01

# --- Procesar argumentos ---
if len(sys.argv) < 3:
  sys.exit("Uso: python " + sys.argv[0] + " x y [--anim | --noanim]")
objetivo=[float(sys.argv[1]), float(sys.argv[2])]
modo_anim = "--anim" in sys.argv

if modo_anim:
  print("🟢 Modo animación activado.")
else:
  print("⚪ Modo estático (una figura por iteración).")

O = cin_dir(th,a)
#O = zeros(len(th) + 1) # Reservamos estructura en memoria
# Calculamos la posicion inicial
print ("- Posicion inicial:")
muestra_origenes(O)

dist = float("inf")
prev = 0.
iteracion = 1
while (dist > EPSILON and abs(prev-dist) > EPSILON/100.):
  prev = dist
  O = [cin_dir(th,a)]
  print(O[0][2][0])
  # Para cada combinación de articulaciones:
  for i in range(len(th) - 1, -1, -1):
    # cálculo de la cinemática inversa:
    chain = cin_dir(th,a)
    efector = chain[-1]
    articulacion = chain[i]
    
    # Comprobar si es prismática (1) o rotatoria (0)
    if (joint_type[i] == 0): 
      alpha_objective = atan2((objetivo[1] - articulacion[1]), (objetivo[0] - articulacion[0]))
      alpha_ef = atan2((efector[1] - articulacion[1]), (efector[0] - articulacion[0]))
      inc_theta = alpha_objective - alpha_ef; 
      
      th[i] = th[i] + inc_theta
      th[i] = normalize(th[i])
      
      # Ver límites
      if (th[i] > upper_limits[i]):
        th[i] = upper_limits[i]
      elif (th[i] < lower_limits[i]):
        th[i] = lower_limits[i]
      
    elif (joint_type[i] == 1):
      u_x = objetivo[0] - efector[0]
      u_y = objetivo[1] - efector[1]
      
      omega = sum(th[:i+1])
        
      v_x = cos(omega)
      v_y = sin(omega)
      
      d = u_x * v_x + u_y * v_y
      a[i] = a[i] + d
      
      # Ver límites
      if (a[i] > upper_limits[i]):
        a[i] = upper_limits[i]
      elif (a[i] < lower_limits[i]):
        a[i] = lower_limits[i]   
      
    O.append(cin_dir(th,a))

  dist = np.linalg.norm(np.subtract(objetivo, O[-1][-1]))
  print ("\n- Iteracion " + str(iteracion) + ':')
  muestra_origenes(O[-1])
  muestra_robot(O, objetivo, modo_anim)
  print ("Distancia al objetivo = " + str(round(dist,5)))
  iteracion += 1
  O[0] = O[-1]
  
if modo_anim:
  plt.ioff()
  plt.show()

# -------- Resultados finales --------
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
