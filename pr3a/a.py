#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Robótica Computacional - CCD con articulaciones rotacionales y prismáticas
# Versión basada en el código proporcionado en clase + mejoras menores

import sys
from math import *
import numpy as np
import matplotlib.pyplot as plt
import colorsys as cs

# ============================================================
# Funciones auxiliares
# ============================================================

def normalize(angle):
    """Normaliza ángulos a (-pi, pi]."""
    while angle > pi: angle -= 2*pi
    while angle <= -pi: angle += 2*pi
    return angle

def clamp(x, lo, hi):
    """Limita x al rango [lo, hi]."""
    return max(lo, min(hi, x))

def muestra_origenes(O, final=None):
    print("Origenes de coordenadas:")
    for i, p in enumerate(O):
        print(f"(O{i})0 =", [round(float(v),3) for v in p])
    if final is not None:
        print("E.Final =", [round(float(v),3) for v in final])

def muestra_robot(O, obj, anim=False):
    if anim:
        plt.clf()
    else:
        plt.figure()

    plt.xlim(-L, L)
    plt.ylim(-L, L)
    plt.grid(True)

    T = [np.array(o).T.tolist() for o in O]
    for i in range(len(T)):
        plt.plot(T[i][0], T[i][1], "-o", color=cs.hsv_to_rgb(i/float(len(T)),1,1))

    plt.plot(obj[0], obj[1], "k*", markersize=12)

    if anim:
        plt.pause(0.25)
    else:
        plt.show()
        plt.close()

def matriz_T(d, th, a, al):
    return [[cos(th), -sin(th)*cos(al),  sin(th)*sin(al), a*cos(th)],
            [sin(th),  cos(th)*cos(al), -sin(al)*cos(th), a*sin(th)],
            [0,        sin(al),          cos(al),         d],
            [0,              0,                0,         1]]

def cin_dir(th, a):
    T = np.identity(4)
    o = [[0.0, 0.0]]
    for i in range(len(th)):
        T = np.dot(T, matriz_T(0, th[i], a[i], 0))
        p = np.dot(T, [0,0,0,1])
        o.append([float(p[0]), float(p[1])])
    return o

# ============================================================
# Configuración del robot
# ============================================================

th = [0.0, 0.0, 0.0, 0.0]
a  = [5.0, 5.0, 5.0, 5.0]

# 0 = rotacional, 1 = prismática
joint_type = [0, 1, 0, 0]

upper_limits = [180, 10, 180, 180]
lower_limits = [-179, 1, -180, -180]

# Convertir límites rotacionales a radianes
for i in range(len(th)):
    if joint_type[i] == 0:
        upper_limits[i] = normalize(upper_limits[i] * pi / 180)
        lower_limits[i] = normalize(lower_limits[i] * pi / 180)

L = sum(a)
EPSILON = 0.01

# ============================================================
# Procesar argumentos de ejecución
# ============================================================

if len(sys.argv) < 3:
    sys.exit("Uso: python prog.py x y [--anim]")

objetivo = [float(sys.argv[1]), float(sys.argv[2])]
modo_anim = "--anim" in sys.argv

print("Modo animación activado." if modo_anim else "Modo estático.")

# ============================================================
# Bucle CCD
# ============================================================

O = cin_dir(th, a)
print("- Posición inicial:")
muestra_origenes(O, objetivo)

dist = float("inf")
prev_dist = 0
iteracion = 1

if modo_anim:
    plt.ion()

while dist > EPSILON and abs(prev_dist - dist) > EPSILON/100:
    prev_dist = dist
    O = [cin_dir(th,a)]

    for j in range(len(th)-1, -1, -1):
        chain = cin_dir(th, a)
        pj = chain[j]
        pe = chain[-1]

        # ROTACIONAL
        if joint_type[j] == 0:
            ang_obj = atan2(objetivo[1]-pj[1], objetivo[0]-pj[0])
            ang_pe  = atan2(pe[1]-pj[1], pe[0]-pj[0])

            delta = normalize(ang_obj - ang_pe)
            th[j] = normalize(th[j] + delta)

            th[j] = clamp(th[j], lower_limits[j], upper_limits[j])

        # PRISMÁTICA
        else:
            dx = objetivo[0] - pe[0]
            dy = objetivo[1] - pe[1]

            omega = sum(th[:j+1])
            vx, vy = cos(omega), sin(omega)

            d = dx*vx + dy*vy
            a[j] += d
            a[j] = clamp(a[j], lower_limits[j], upper_limits[j])

        O.append(cin_dir(th, a))

    dist = np.linalg.norm(np.array(objetivo) - np.array(O[-1][-1]))

    print(f"\n- Iteración {iteracion}, dist =", round(dist,4))
    muestra_origenes(O[-1])
    muestra_robot(O, objetivo, anim=modo_anim)

    iteracion += 1

if modo_anim:
    plt.ioff()
    plt.show()

print("\nRESULTADOS FINALES:")
print("Iteraciones:", iteracion)
print("Distancia final:", dist)
print("Ángulos finales:", [round(th[i],3) for i in range(len(th))])
print("Longitudes finales:", [round(a[i],3) for i in range(len(a))])
