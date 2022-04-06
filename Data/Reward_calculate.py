import sys

# lines_after_17 = f.readlines()[17:]


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('no argument')
        sys.exit()
    py_text = sys.argv[1]
    print(py_text)
    f = open(py_text, 'r')
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
                print("changing.")

            if num_start is True:
                if i == " ":
                    num_start = False
                    actual_num = int(temp)
                    #print("num: ", actual_num)
                    p_start = True
                    temp = ""
                else:
                    temp += i
            if p_start is True:
                if i == "]":
                    p_start = False
                    actual_pri = float(temp)
                    #print("pri: ", actual_pri)
                    temp = ""
                    num_list[iteration_count][actual_num] += actual_pri
                else:
                    temp += i

            if i == "[":
                num_start = True

            else:
                continue

    f.close()

    for i in range(15):
        print("map", i, ": ", num_list[0][i])
    print()
    for i in range(15):
        print("map", i, ": ", num_list[1][i])
