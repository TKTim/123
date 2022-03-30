import sys
from numpy import random


# lines_after_17 = f.readlines()[17:]

def getBigmap(i):
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
    return ans


def getReward(num_list, output):
    for i in range(1000):
        reward_temp = 0
        for j in range(625):  # From map j in time i
            if output[i][j] == 1:  # Catched by Small m
                continue
            else:
                if output[i][625 + getBigmap(j)] == 1:  # Catched by Big M
                    reward_temp += -1.0 * num_list[i][j]
                else:  # Fail, and fetching from data set
                    reward_temp += -10.0 * num_list[i][j]

        output[i][794] = reward_temp


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

    num_list = [[0 for _ in range(625)] for _ in range(1000)]  # numbers of maps
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
                if i == " ":  # map_num is collected
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

    # Random state:795 =  794 maps + 1 target_value
    output = [[0 for _ in range(795)] for _ in range(1000)]
    # x = random.choice([0, 1], p=[0.5, 0.5], size=(100))
    random.seed(10)
    for i in range(1000):
        # decision variables {0,1}
        constrant_maps = 200
        for j in range(794):
            rr = random.randint(0, 2)
            if constrant_maps <= 0:
                output[i][j] = 0  # Have changed
            else:
                if rr == 1:
                    output[i][j] = 1
                    constrant_maps -= 1
                else:
                    output[i][j] = 0  # Have changed

    getReward(num_list, output)

    for i in range(1000):
        print(" iter: ", i, file=fp)
        print("[", end="", file=fp)
        for j in range(795):
            print(output[i][j], " ", end="", file=fp)
        print("]", file=fp)

    fp.close()

'''
    for i in range(15):
        print("map", i, ": ", num_list[0][i])
'''
