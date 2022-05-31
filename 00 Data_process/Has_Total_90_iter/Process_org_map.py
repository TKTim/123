import sys

py_text = sys.argv[1]
py_output = sys.argv[2]

f = open(py_text, 'r')
py_out = open(py_output, "w+")

small_number = 1010

'''
# Map formation #
'''
lines = (line.rstrip() for line in f.readlines())  # All lines including the blank ones. Skip first line.
lines = (line for line in lines if line)  # Non-blank lines


map_ = [[0 for _ in range(3)] for _ in range(small_number)]
timer = 0
for line in lines:
    # print(line)
    temp = line.split(' ')
    for i in temp:
        temp_in = i.split(',')
        map_[int(temp_in[1])][0] = int(temp_in[0])  # temp_in[0]:index, temp_in[1]: Big map index
        map_[int(temp_in[1])][1] = int(temp_in[1])
        map_[int(temp_in[1])][2] = int(temp_in[2])
        # print(int(temp_in[0]), int(temp_in[1]), int(temp_in[2]))


for i in range(small_number):
    print("{},{} ".format(map_[i][1], map_[i][2]),end="", file=py_out)


f.close()
py_out.close()