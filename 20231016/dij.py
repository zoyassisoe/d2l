def dijkstra(E, n0):
    v_size = len(E)
    l=[1e5]*v_size
    l[n0]=0
    s = [n0]
    s_hat = [i for i in range(v_size)]
    s_hat.remove(n0)
    parent = [-1]*v_size

    while s_hat:
        d = 1e5
        index_u, index_v = -1, -1
        for u in s:
            for v in s_hat:
                d_uv = l[u]+E[u][v]
                if d > d_uv:
                    d = d_uv
                    index_u = u
                    index_v = v
        l[index_v]=d
        parent[index_v] = index_u
        s.append(index_v)
        s_hat.remove(index_v)
        print('next')
        print(s, s_hat)
        print(l, d, index_v)
        print(index_u, index_v)
    
    return l, parent


if __name__=='__main__':
    E = [[0,5,2,1e5],[5,0,7,3],[2,7,0,1],[1e5,3,1,0]]

    l,p = dijkstra(E,1)
    print(l, p)