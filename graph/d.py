
def is_graph(ds:list):
    while len(ds)>1:
        ds.sort(reverse=True)
        d0 = ds[0]
        ds.pop(0)
        for i in range(d0):
            ds[i] -= 1
    return ds

print(is_graph([5,4,3,3,2,2,2,1,1,1]))
print(is_graph([6,6,5,4,3,3,1]))
print(is_graph([2,2,2,0]))          
print(is_graph([3,3,3,3,0]))  
print(is_graph([2,1,1,0]))  
# [0]
# [-1]
# [0]
# [0]
# [0]