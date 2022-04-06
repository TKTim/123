import sys
from numpy import random

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('no argument')
        sys.exit()
    py_text = sys.argv[1]
    py_write = sys.argv[2]

    print(py_text)

    f = open(py_text, 'r')
    fp = open(py_write, "w+")

    lines = (line.rstrip() for line in f.readlines())  # All lines including the blank ones. Skip first line.
    lines = (line for line in lines if line)  # Non-blank lines

    num_list = [[0 for _ in range(333)] for _ in range(200)]  # numbers of maps
    iteration_count = -1
    for line in lines:
        temp = ""
        num_start = False
        p_start = False
        actual_num = 0
        actual_pri = 0.0
        for i in line:

            if i == "T":
                iteration_count += 1

            if num_start is True:
                if i == " ":
                    num_start = False
                    actual_num = int(temp)
                    # print("num: ", actual_num)
                    p_start = True
                    temp = ""
                else:
                    temp += i
            if p_start is True:
                if i == "]":
                    p_start = False
                    actual_pri = float(temp)
                    # print("pri: ", actual_pri)
                    temp = ""
                    num_list[iteration_count][actual_num] += actual_pri
                else:
                    temp += i

            if i == "[":
                num_start = True

            else:
                continue

    # Close Files
    f.close()

    for i in range(200):
        print(" iter: ", i, file=fp)
        print("[", end="", file=fp)
        for j in range(333):
            print(num_list[i][j], end=" ", file=fp)
        print("]", file=fp)

    fp.close()

'''
    for i in range(15):
        print("map", i, ": ", num_list[0][i])
'''
