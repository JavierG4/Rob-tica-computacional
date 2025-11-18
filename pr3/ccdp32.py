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
  print('Origenes de coordenadas:')
  for i in range(len(O)):
    print('(O'+str(i)+')0\t= '+str([round(j,3) for j in O[i]]))
  if final:
    print('E.Final = '+str([round(j,3) for j in final]))

def muestra_robot(O,obj,anim=False):
  """
  Si anim=False, muestra cada iteración en una figura estática.
  Si anim=True, actualiza una única ventana (animación).
  """
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
  plt.plot(obj[0], obj[1], 'k*', markersize=10)
  plt.title("Iteración CCD")
  plt.xlabel("X")
  plt.ylabel("Y")

  if anim:
    plt.pause(0.3)
  else:
    plt.show()
    plt.close()

def matriz_T(d,th,a,al):
  return [[cos(th), -sin(th)*cos(al),  sin(th)*sin(al), a*cos(th)],
          [sin(th),  cos(th)*cos(al), -sin(al)*cos(th), a*sin(th)],
          [0,        sin(al),          cos(al),         d],
          [0,              0,                0,         1]]

def cin_dir(th,a):
  T = np.identity(4)
  o = [[0,0]]
  for i in range(len(th)):
    T = np.dot(T,matriz_T(0,th[i],a[i],0))
    tmp=np.dot(T,[0,0,0,1])
    o.append([tmp[0],tmp[1]])
  return o

# ******************************************************************************
# Cálculo de la cinemática inversa de forma iterativa por el método CCD

# valores iniciales
th=[0.,0.,0.]
a =[5.,5.,5.]
L = sum(a)
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

O=cin_dir(th,a)
print ("- Posicion inicial:")
muestra_origenes(O)

dist = float("inf")
prev = 0.
iteracion = 1

if modo_anim:
  plt.ion()  # activar modo interactivo

while (dist > EPSILON and abs(prev-dist) > EPSILON/100.):
  prev = dist
  O=[cin_dir(th,a)]

  # Para cada articulación:
  for j in range(len(th)-1, -1, -1):
    chain = cin_dir(th,a)
    pj = np.array(chain[j]) # origen de la j-ésima articulación
    pe = np.array(chain[-1]) # extremo del efector final
    # Vectores desde la j-ésima articulación hasta:
    # - el extremo del efector final
    # - el objetivo
    r1 = pe - pj
    r2 = np.array(objetivo) - pj

    if np.linalg.norm(r1) < 1e-9 or np.linalg.norm(r2) < 1e-9:
      O.append(chain)
      continue

    # Calcular el ángulo entre r1 y r2
    cross = r1[0]*r2[1] - r1[1]*r2[0]
    dot = float(r1.dot(r2))
    delta = atan2(cross, dot)
    th[j] += delta

    # Normalizar ángulo entre -pi y pi
    th[j] = (th[j] + pi) % (2*pi) - pi

    O.append(cin_dir(th,a))

  dist = np.linalg.norm(np.subtract(objetivo,O[-1][-1]))
  print ("\n- Iteracion " + str(iteracion) + ':')
  muestra_origenes(O[-1])
  muestra_robot(O,objetivo,anim=modo_anim)
  print ("Distancia al objetivo = " + str(round(dist,5)))
  iteracion+=1
  O[0]=O[-1]

if modo_anim:
  plt.ioff()
  plt.show()

# -----------------------------------------------------------------------------
# Resultados finales
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
