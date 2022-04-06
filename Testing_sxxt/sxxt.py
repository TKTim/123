import torch as t
import os

i = 200
loc_temp = int(i / 50)
line_temp = int(i / 100)
loc = int(i - 50 * loc_temp)
map_loc = int(loc / 2)
ans = int(map_loc + 25 * line_temp)

print(ans)
