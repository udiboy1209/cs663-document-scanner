from random import sample
from itertools import combinations

def area_tri(p):
    x = [k[0] for k in p]
    y = [k[1] for k in p]

    a = 0.5*abs((x[1]-x[0])*(y[2]-y[0])-(y[1]-y[0])*(x[2]-x[0]))
    return a

def measure_collinearity(p):
    p_combs = combinations(p,3)
    d = min([area_tri(k) for k in p_combs])
    return d

def choose_points(points,N,thres):
    n = 0
    chosen_points = []
    while n < N:
        new_choice = sample(points,4)
        new_choice = sorted(new_choice, key=lambda p: p[0])
        d = measure_collinearity(new_choice)
        if d > thres and new_choice not in chosen_points:
            chosen_points.append(new_choice)
            n += 1

    return chosen_points

if __name__ == '__main__':
    print(choose_points([(1,1),(2,3),(3,3),(5,0),(0,10),(10,0),(0,0)],5,0.1))
