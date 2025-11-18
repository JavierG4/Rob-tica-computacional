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
import math

PI = math.pi

# *******************************************************************************
# Declaración de funciones (manteniendo nombres originales cuando es posible)
# *******************************************************************************

def muestra_origenes(O,final=0):
  print('Origenes de coordenadas:')
  for i in range(len(O)):
    coords = O[i]
    if hasattr(coords, "tolist"):  # Si es numpy array
      coords = coords.tolist()
    if isinstance(coords, (list, tuple, np.ndarray)):
      coords_str = ', '.join(f"{float(j):.3f}" for j in coords)
    else:
      coords_str = str(coords)
    print(f"(O{i})0\t= {coords_str}")
  if final is not None and final != 0:
    if hasattr(final, "tolist"):
      final = final.tolist()
    if isinstance(final, (list, tuple, np.ndarray)):
      final_str = ', '.join(f"{float(j):.3f}" for j in final)
    else:
      final_str = str(final)
    print(f"E.Final = {final_str}")

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
  # Se mantiene por compatibilidad aunque no se usa en la cinemática 2D simplificada
  return [[cos(th), -sin(th)*cos(al),  sin(th)*sin(al), a*cos(th)],
          [sin(th),  cos(th)*cos(al), -sin(al)*cos(th), a*sin(th)],
          [0,        sin(al),          cos(al),         d],
          [0,              0,                0,         1]]

# ---------- Nueva cinemática directa 2D que respeta prismáticas ---------------
def cin_dir(th, a, prismatica):
  """
  Calcula orígenes de cada articulación (lista de [x,y]) para una cadena plana 2D.
  - th: lista de ángulos (radianes). Para articulaciones prismáticas su valor se ignora.
  - a: lista de longitudes / desplazamientos prismáticos.
  - prismatica: lista booleana indicando si la articulación i es prismática (True) o rotacional (False).
  Retorna lista de orígenes: O[0] = base, O[1] = origen art1, ..., O[n] = efector final.
  """
  n = len(th)
  if not (len(a) == n == len(prismatica)):
    raise ValueError("th, a y prismatica deben tener la misma longitud")

  origins = [[0.0, 0.0]]
  pos = np.array([0.0, 0.0])
  current_angle = 0.0  # orientación global acumulada (radianes)

  for i in range(n):
    if prismatica[i]:
      # Traslación a lo largo del eje X local por a[i]
      dx_local = np.array([a[i], 0.0])
      c = math.cos(current_angle); s = math.sin(current_angle)
      R = np.array([[c, -s],[s, c]])
      disp = R.dot(dx_local)
      pos = pos + disp
      # ángulo no cambia
    else:
      # Rotamos por th[i] y luego avanzamos a lo largo del eje X local por a[i]
      current_angle += th[i]
      c = math.cos(current_angle); s = math.sin(current_angle)
      R = np.array([[c, -s],[s, c]])
      dx_local = np.array([a[i], 0.0])
      disp = R.dot(dx_local)
      pos = pos + disp
    origins.append([float(pos[0]), float(pos[1])])

  return origins

# ---------- Interprete valores numéricos respetando tipo de articulación -----
def interpretar_valor_num(v, es_rotacional=True):
  """
  Interpreta v como número. Si es_rotacional==True: acepta grados o radianes.
  Heurística: si abs(v) > 2*pi se asume grados y se convierte a radianes.
  Si es_rotacional==False (prismática) se devuelve float(v) sin conversión.
  """
  try:
    vv = float(v)
  except Exception:
    # en caso de sufijos "deg" o "rad"
    s = str(v).strip().lower()
    if s.endswith("deg"):
      return math.radians(float(s[:-3]))
    if s.endswith("rad"):
      return float(s[:-3])
    raise ValueError("Valor no numérico: {}".format(v))

  if es_rotacional:
    if abs(vv) > 2 * math.pi:
      return radians(vv)   # asumimos que dio grados grandes -> convertir
    return vv
  else:
    # prismática: no convertir, son longitudes
    return vv

def clamp(x, lo, hi):
  return max(lo, min(hi, x))

# *******************************************************************************
# Cálculo de la cinemática inversa de forma iterativa por el método CCD
# *******************************************************************************

# valores iniciales (conservados nombres originales)
th=[0.,0., 0., 0.]
a =[5.,5.,5., 5.]

# si son prismaticas o no
prismatica = [False, True, False, False]

# Límites: para prismáticas son límites de longitud, para rotacionales son límites de ángulo
# Nota: aquí interpretamos cada límite según el tipo de articulación correspondiente.
# Antes se aplicaba interpretar_valor_num a todos indiscriminadamente; eso convertía longitudes erróneamente.
# Ajusta estos valores si quieres límites distintos (puedes usar strings "90deg" o números).
raw_tMin = [-179, 1, -180, -180]
raw_tMax = [180, 10, 180, 180]

tMin = []
tMax = []
for i in range(len(raw_tMin)):
  if prismatica[i]:
    # prismatic: tratar como longitudes (no convertir grados->rad)
    tMin.append(interpretar_valor_num(raw_tMin[i], es_rotacional=False))
    tMax.append(interpretar_valor_num(raw_tMax[i], es_rotacional=False))
  else:
    # rotacional: interpretar grados/radianes
    tMin.append(interpretar_valor_num(raw_tMin[i], es_rotacional=True))
    tMax.append(interpretar_valor_num(raw_tMax[i], es_rotacional=True))

L = sum(a) + 1  # Ajustar límite de visualización considerando extensión máxima
EPSILON = .01

# --- Procesar argumentos ---

if len(sys.argv) < 3:
  sys.exit("Uso: python " + sys.argv[0] + " x y [--anim] [--noanim] [--solo-final]")
objetivo=[float(sys.argv[1]), float(sys.argv[2])]
modo_anim = "--anim" in sys.argv
modo_noanim = "--noanim" in sys.argv
solo_final = "--solo-final" in sys.argv

if modo_anim:
  print("🟢 Modo animación activado.")
elif modo_noanim:
  print("⚪ Modo estático (una figura por iteración).")
if solo_final:
  print("🔵 Solo se mostrará la última gráfica, pero se mostrarán todos los datos de iteración.")

O=cin_dir(th,a,prismatica)
print ("- Posicion inicial:")
muestra_origenes(O)

dist = float("inf")
prev = 0.
iteracion = 1

if modo_anim:
  plt.ion()  # activar modo interactivo

graficas = []
# Condición de parada: distancia y cambio pequeño entre iteraciones
while (dist > EPSILON and abs(prev-dist) > EPSILON/100.):
  prev = dist
  O=[cin_dir(th,a,prismatica)]

  # Para cada articulación (de la última a la primera):
  for j in range(len(th)-1, -1, -1):
    chain = cin_dir(th,a,prismatica)
    pj = np.array(chain[j]) # origen de la j-ésima articulación
    pe = np.array(chain[-1]) # extremo del efector final
    r1 = pe - pj
    r2 = np.array(objetivo) - pj

    if np.linalg.norm(r1) < 1e-9 or np.linalg.norm(r2) < 1e-9:
      O.append(chain)
      continue

    if prismatica[j]:
      # omega = suma de ángulos hasta la articulación j (EXCLUYENDO la j-ésima)
      omega = 0.0
      for k in range(0, j):
        if not prismatica[k]:
          omega += th[k]
      # eje local X en coordenadas globales
      u = np.array([cos(omega), sin(omega)])

      # Proyección CORRECTA:
      # utilizar el desplazamiento del EFECTOR FINAL (target - posicion_actual_efector)
      # y proyectarlo sobre el eje u para decidir cuánto añadir a a[j]
      disp_end = np.array(objetivo) - pe   # desplazamiento deseado del efector
      d = float(np.dot(u, disp_end))

      L_nueva = a[j] + d
      # clamp a límites prismáticos (tMin/tMax contienen límites apropiados para prismática)
      a[j] = max(tMin[j], min(tMax[j], L_nueva))
    else:
      # Rotacional: cálculo del ángulo entre r1 y r2 (signo incluido)
      cross = r1[0]*r2[1] - r1[1]*r2[0]
      dot = float(r1.dot(r2))
      delta = atan2(cross, dot)
      th[j] += delta
      # normalizar entre -pi y pi
      th[j] = (th[j] + pi) % (2*pi) - pi
      # recortar a límites angulares (tMin/tMax contienen límites para rotacionales)
      if th[j] < tMin[j]:
        th[j] = tMin[j]
      elif th[j] > tMax[j]:
        th[j] = tMax[j]
    O.append(cin_dir(th,a,prismatica))

  dist = np.linalg.norm(np.subtract(objetivo,O[-1][-1]))
  print ("\n- Iteracion " + str(iteracion) + ':')
  muestra_origenes(O[-1])
  if not solo_final:
    muestra_robot(O,objetivo,anim=modo_anim)
  else:
    # Solo guardar la última para graficar después
    graficas = [[ [p[:] for p in O], objetivo[:] ]]
  print ("Distancia al objetivo = " + str(round(dist,5)))
  iteracion+=1
  O[0]=O[-1]

if solo_final and graficas:
  # Mostrar solo la última gráfica
  O_final, obj_final = graficas[-1]
  muestra_robot(O_final, obj_final, anim=False)

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
for i in range(len(a)):
  print ("  L" + str(i+1) + "     = " + str(round(a[i],3)))
