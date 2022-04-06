import sys

def getBigmap(i):
    print("i: ", i)
    if i % 25 == 0:
        i = i + 1
    loc_temp = int(i / 25)
    line_temp = int(i / 49)
    loc = int(i - 25 * loc_temp)
    if loc <= 0:
        ans = int(i / 2)
    else:
        map_loc = int(loc / 2)
        ans = int(map_loc + 13 * line_temp)
    print("loc_temp: ", loc_temp)
    print("line_temp: ", line_temp)
    print("loc: ", loc)
    print("map_loc: ", map_loc)
    return ans


for i in [2, 5, 12, 13, 16]:
    print("ans: ", 625+getBigmap(i), "\n --------")
