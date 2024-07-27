#I decided to bite the bullet and use sets, since set inclusion is supposedly quicker 
#than array inclusion, also im a dirty mathematician who understands sets better

import itertools
from itertools import product
import random
import numpy as np
import math as m
import time
import matplotlib.pyplot as plt


def power_set(A):
    length = len(A)
    return {
        frozenset({e for e, b in zip(A, f'{i:{length}b}') if b == '1'})
        for i in range(2 ** length)
    }


def flatten(tup):
    if type(tup) != tuple:
        return tup
    return tuple(itertools.chain(*([x] if not isinstance(x, tuple) else x for x in tup)))    

def ex(n, d, L, draw = False, timeCheck = False):
    #print()
    #print("Setup: n="+str(n)+", d="+str(d)+", L="+str(L))
    
    if(timeCheck):
       T = time.time()

    two = set((0,1))
    
    space = frozenset(product(two, repeat = n))
    
    X = set(range(n))
    
    PX = power_set(X)
    
    Q = set()
    
    #f = '#0'+str(d)+'b'
    
    for SET in PX:  
        if len(SET) == d: #We have a set of positions to put the 2s
            for dispersal in range(0,2**(n-d)): #We choose a number to represent in binary which are 1s and 0s
                                                #For example for n = 6, {0,2,3}, 5=101 gives 2 x 1 x 2 x 2 x 0 x 1
                iterate = 0
                D = list(bin(dispersal)[2:].zfill(n-d))
                K = set()
                if 0 in SET:
                    S = set(two)
                else:
                    S = set({int(D[iterate])})
                    iterate += 1
                for s in S:
                    K.add(flatten(s))           #All these flattenings are because product(product) gives coordinates ((a,b),c) but we want (a,b,c)
    
                for i in range(1,n):            
                    if i in SET:
                        S = product(K, two)
                    else:
                        S = product(K, set({int(D[iterate])}))
                        iterate += 1
                    for s in S:
                        K.add(flatten(s))
                J = set()
                for k in K:
                    if type(k) != int:
                        if len(k) == n:
                            J.add(k)                #Since i never really clear K we still have the small coordinates but the top dimensional coordinates are the only ones we want
    
                Q.add(frozenset(J))
    
    m = 0

    if(timeCheck):    
        #print("-----------------------------Boxes Made----------------------------")
        #print("This took ",time.time() - T, "s")
        T = time.time()
    
    #powerset = list(power_set(space))
    
    #print("----------------------------Powerset Made--------------------------")
    #print("This took ",time.time() - T, "s")
    #T = time.time()

    # setsMade = 0
    # intersectionsDone = 0
    # setsSkipped = 0

    K = []

    P = tuple(np.zeros(n, dtype=int))

    s = list(space)

    s.remove(P)

    x = len(s)
    
    for i in range(1 << x):                                  #This is a very straightforward checking of C against all the boxes
        C = [s[j] for j in range(x) if (i & (1 << j))]
        C.append(P)
        #setsMade += 1
        t = True
        l = len(C)
        if(l <= m):
          #setsSkipped += 1
          continue

        for box in Q:
            if len(set(C) & box) > L:
                t = False
                continue
            #intersectionsDone += 1
        
        if (t and l >= m):
            K = C                                         #We keep K just so we can draw it at the end
            m = l
    
    if(timeCheck):
        print("----------------------------Code Checked---------------------------")
        print("This took ",time.time() - T, "s")
        T = time.time()
    
    # print("Sets: "+str(setsMade))
    # print("Intersections: "+str(intersectionsDone))
    # print("Sets Skipped "+str(setsSkipped))
    
    # print("Number of intersections should be "+str((setsMade-setsSkipped) * len(Q)))


    # value = np.ceil(float(2)**(n+1)/float(3))
    # val   = str(int(value))
    
    # print("============================================")
    # print("Computed Value: "+str(m))
    # print("============================================")
    # print("The Value from proof: "+val)
    # print("============================================")
    # print()
    print("ex("+str(n)+","+str(d)+","+str(L)+")="+str(m))
    # print("An example code would be "+str(set(K)))
    if(draw):
        drawCode(K, n)

# for n in range(1,10):
#     for L in range(1,2**n):
#         ex(n+1,n,L)
#     print("-----------")

# Drawing the Code, honestly more of a pain than i was expecting
# going from point in 2^n to point in R^2 is annoying

def drawCube(n):
  two = set((0,1))
    
  space = frozenset(product(two, repeat = n))

  for point1 in space:
    for point2 in space:
      if hd(point1,point2) == 1:
        plt.plot([drawPointX(point1, n),drawPointX(point2, n)],
                 [drawPointY(point1, n),drawPointY(point2, n)], "r")

def drawCode(C, n):
  
  drawCube(n)
  
  X = np.array([])
  Y = np.array([])
  for p in C:
      X = np.append(X, drawPointX(p,n))
      Y = np.append(Y, drawPointY(p,n))
  plt.scatter(X,Y)
  plt.show()

def drawPointY(p,n):
    s = 0
    for coord in p:
        s += coord
    return s

def drawPointX(p,n):
    s = 0
    h = drawPointY(p,n)
    for i in range(n):
        s += (2**(i))*((-1)**(1+p[i]+i)) 
    return s
    # return s - h*(h-1)/2 - m.comb(n,h)/2


def hd(p,q):
  s = 0
  for i in range(len(p)):
    s += (p[i]-q[i])**2
  return s

#====================================

print("compute ex(n,d,L), be aware that it's very slow for n>4")
n = 0
d = np.inf
L = np.inf

n = int(input("ex(n,d,L): n = "))
while(d > n):
  d = int(input("ex("+str(n)+",d,L): d = "))
while(L > 2**d):
  L = int(input("ex("+str(n)+","+str(d)+",L): L = "))

ex(n,d,L, draw = True, timeCheck = True)
