import sys

if __name__ == '__main__':

    Big_map_max = 0
    Smol_map_max = 0

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
                if int(temp_in[0]) > Smol_map_max:
                    Smol_map_max = int(temp_in[0])
                if int(temp_in[1]) > Big_map_max:
                    Big_map_max = int(temp_in[1])
                # print(int(temp_in[0]), int(temp_in[1]))
            except:
                print(int(temp_in[0]), "  ", int(temp_in[1]))
                sys.exit()
    ptr_in.close()

print("Big: ", Big_map_max)
print("Smol: ", Smol_map_max)
