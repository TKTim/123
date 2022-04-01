import sys


def getBigmap(i):
    print("i: ", i)
    loc_temp = int(i / 50)
    line_temp = int(i / 100)
    loc = int(i - 50 * loc_temp)
    map_loc = int(loc / 2)
    ans = int(map_loc + 25 * line_temp)
    print("loc_temp: ", loc_temp)
    print("line_temp: ", line_temp)
    print("loc: ", loc)
    print("map_loc: ", map_loc)
    print("ans: ", ans)
    return ans

print(getBigmap(1200))
