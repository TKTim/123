import sys

if __name__ == '__main__':
    py_text = sys.argv[1]

    ptr_in = open(py_text, "r")

    lines = (line.rstrip() for line in ptr_in.readlines())  # All lines including the blank ones. Skip first line.
    lines = (line for line in lines if line)  # Non-blank lines

    map_ = [0]*2161  # 0~2160

    for line in lines:
        # print(line)
        temp = line.split(' ')
        for i in temp:
            temp_in = i.split(',')
            try:
                map_[int(temp_in[0])] = int(temp_in[1])  # temp_in[0]:index, temp_in[1]: Big map index
                # print(int(temp_in[0]), int(temp_in[1]))
            except:
                print(int(temp_in[0]), "  ", int(temp_in[1]))
                sys.exit()
    ptr_in.close()

    map_max = 0
    # Get the big map max
    for i in range(len(map_)):
        # print("i: {}, map_: {}".format(i, map_[i]))
        if map_[i] > map_max:
            map_max = map_[i]

    print("map_max: ", map_max)
