#! /usr/bin/env python
# -- coding: utf-8 --

# Robótica Computacional - 
# Grado en Ingeniería Informática (Cuarto)
# Práctica: Resolución de la cinemática inversa mediante CCD
#           (Cyclic Coordinate Descent).

import sys
from math import *
import numpy as np
import matplotlib.pyplot as plt
import colorsys as cs
import math

PI = math.pi

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


# ----------------------------------------------
# VALORES INICIALES DE LAS ARTICULACIONES
# ----------------------------------------------

th = [0., 0., 0.]        # Ángulos iniciales de cada articulación (en radianes).
                         # Para articulaciones prismáticas este valor no se usa como ángulo
                         # pero se mantiene por la cinemática directa.

a = [5., 5., 5.]         # Longitudes iniciales de cada eslabón.
                         # Si una articulación es prismática, su longitud a[j] cambiará durante el CCD.

# ----------------------------------------------
# TIPO DE ARTICULACIONES
# True  = prismática (la longitud cambia)
# False = rotacional (el ángulo cambia)
# ----------------------------------------------

prismatica = [False, True, False]

# ----------------------------------------------
# LÍMITES
# Para prismáticas: límites de longitud (mín y máx)
# Para rotacionales: límites de ángulo (mín y máx)
# ----------------------------------------------

tMin = [ -30.0 * pi / 180, 1,-30.0 * pi / 180]  # Primera: longitud mínima 1.0, resto ángulos -30°
tMax = [ 30.0 * pi / 180, 10,30.0 * pi / 180]   # Primera: longitud máxima 10.0, resto ángulos 30°

L = sum(a) + 5  # Para la visualización del robot
EPSILON = .01   # Umbral para detener la iteración CCD

# ----------------------------------------------
# PROCESAR ARGUMENTOS DE ENTRADA
# ----------------------------------------------

if len(sys.argv) < 3:
  sys.exit("Uso: python " + sys.argv[0] + " x y [--anim | --noanim]")

objetivo = [float(sys.argv[1]), float(sys.argv[2])]  # Punto objetivo (x,y)
modo_anim = "--anim" in sys.argv                    # Modo animación opcional

if modo_anim:
  print("🟢 Modo animación activado.")
else:
  print("⚪ Modo estático (una figura por iteración).")

# ----------------------------------------------
# Cálculo inicial de la cinemática directa
# ----------------------------------------------

O = cin_dir(th, a)  # Orígenes de todas las articulaciones
print("- Posicion inicial:")
muestra_origenes(O)

dist = float("inf")  # Distancia inicial al objetivo
prev = 0.
iteracion = 1

if modo_anim:
  plt.ion()

# ----------------------------------------------
# BUCLE PRINCIPAL CCD
# ----------------------------------------------

while (dist > EPSILON and abs(prev - dist) > EPSILON/100.):
  prev = dist
  O = [cin_dir(th, a)]  # Recalcula cinemática actual

  # --------------------------------------------------------------
  # RECORRER ARTICULACIONES DESDE LA ÚLTIMA HACIA ATRÁS (CCD)
  # --------------------------------------------------------------

  for j in range(len(th)-1, -1, -1):
    chain = cin_dir(th, a)        # Cinemática actualizada
    pj = np.array(chain[j])       # Posición de la articulación j
    pe = np.array(chain[-1])      # Posición del efector final

    # Vectores importantes:
    r1 = pe - pj                  # Vector desde articulación j → efector final
    r2 = np.array(objetivo) - pj  # Vector desde articulación j → objetivo

    # Evitar divisiones por cero
    if np.linalg.norm(r1) < 1e-9 or np.linalg.norm(r2) < 1e-9:
      O.append(chain)
      continue

    # ==============================================================
    # CASO 1: ARTICULACIÓN PRISMÁTICA → AJUSTAR LONGITUD
    # ==============================================================
    if prismatica[j]:

      # -------------------------------
      # Calcular la dirección del eslabón j
      # -------------------------------

      if j > 0:
        # Direccion = posición(j) - posición(j-1)
        pj_prev = np.array(chain[j-1])
        dir_eslabon = pj - pj_prev

        # Normalizar la dirección
        if np.linalg.norm(dir_eslabon) > 1e-9:
          dir_eslabon = dir_eslabon / np.linalg.norm(dir_eslabon)
        else:
          # Si la dirección es indeterminada, usar orientación acumulada
          dir_eslabon = np.array([cos(sum(th[:j])), sin(sum(th[:j]))])
      else:
        # Para j = 0: el eslabón no tiene articulación anterior
        # Se usa la orientación base (ángulos anteriores sumados)
        dir_eslabon = np.array([cos(sum(th[:j])), sin(sum(th[:j]))])

      # -------------------------------
      # PROYECCIÓN DE r2 EN LA DIRECCIÓN DEL ESLABÓN
      # Indica cuánto "conviene" extender o contraer el eslabón
      # -------------------------------
      proyeccion = r2.dot(dir_eslabon)

      # Diferencia entre:
      # - Lo que deberíamos extendernos hacia el objetivo (proyección)
      # - La longitud real hacia el efector final (‖r1‖)
      delta = proyeccion - np.linalg.norm(r1)

      # -------------------------------
      # AJUSTAR LONGITUD DEL ESLABÓN PRISMÁTICO
      # -------------------------------
      a[j] += delta

      # -------------------------------
      # LIMITAR LA LONGITUD A SU RANGO FÍSICO
      # -------------------------------
      if a[j] < tMin[j]: a[j] = tMin[j]
      elif a[j] > tMax[j]: a[j] = tMax[j]

    # ==============================================================
    # CASO 2: ARTICULACIÓN ROTACIONAL → AJUSTAR ÁNGULO
    # ==============================================================
    else:

      # Ángulo mínimo para alinear r1 con r2:
      cross = r1[0]*r2[1] - r1[1]*r2[0]   # Signo del giro (producto cruzado)
      dot = float(r1.dot(r2))            # Coseno del ángulo (producto punto)
      delta = atan2(cross, dot)          # Giro necesario

      # Actualizar ángulo
      th[j] += delta

      # Normalizar entre [-pi, pi]
      th[j] = (th[j] + pi) % (2*pi) - pi

      # Aplicar límites de ángulo
      if th[j] < tMin[j]: th[j] = tMin[j]
      elif th[j] > tMax[j]: th[j] = tMax[j]

    # Guardar la nueva configuración tras mover esta articulación
    O.append(cin_dir(th, a))

  # --------------------------------------------------------------
  # RE-CALCULAR DISTANCIA AL OBJETIVO
  # --------------------------------------------------------------
  dist = np.linalg.norm(np.subtract(objetivo, O[-1][-1]))

  print("\n- Iteracion " + str(iteracion) + ':')
  muestra_origenes(O[-1])
  muestra_robot(O, objetivo, anim=modo_anim)
  print("Distancia al objetivo = " + str(round(dist,5)))

  iteracion += 1
  O[0] = O[-1]

# ----------------------------------------------
# RESULTADOS FINALES
# ----------------------------------------------

if modo_anim:
  plt.ioff()
  plt.show()

if dist <= EPSILON:
  print("\n" + str(iteracion) + " iteraciones para converger.")
else:
  print("\nNo hay convergencia tras " + str(iteracion) + " iteraciones.")

print("- Umbral de convergencia epsilon: " + str(EPSILON))
print("- Distancia al objetivo:          " + str(round(dist,5)))
print("- Valores finales de las articulaciones:")
for i in range(len(th)):
  print("  theta" + str(i+1) + " = " + str(round(th[i],3)))
for i in range(len(a)):
  print("  L" + str(i+1) + "     = " + str(round(a[i],3)))
