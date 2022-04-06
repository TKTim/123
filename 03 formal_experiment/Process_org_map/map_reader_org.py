import sys

def map_read(ptr_in):
    Big_map_max = 0
    Smol_map_max = 0

    lines = (line.rstrip() for line in ptr_in.readlines())  # All lines including the blank ones. Skip first line.
    lines = (line for line in lines if line)  # Non-blank lines

    # map_ = [0] * 5000  # 0~2160

    for line in lines:
        # print(line)
        temp = line.split(' ')
        for i in temp:
            temp_in = i.split(',')
            try:
                # map_[int(temp_in[0])] = int(temp_in[1])
                # temp_in[0]:org index, temp_in[1]:index, temp_in[2]: Big map index
                if int(temp_in[1]) > Smol_map_max:
                    Smol_map_max = int(temp_in[1])
                if int(temp_in[2]) > Big_map_max:
                    Big_map_max = int(temp_in[2])
                # print(int(temp_in[0]), int(temp_in[1]))

            except:
                print(int(temp_in[0]), "  ", int(temp_in[1]))
                sys.exit()
    ptr_in.close()

    Big_map_max += 1
    Smol_map_max += 1
    Total_map = Big_map_max + Smol_map_max
    print("Big: ", Big_map_max)
    print("Smol: ", Smol_map_max)
    print("Total: ", Total_map)

    return Big_map_max


if __name__ == '__main__':
    py_text = sys.argv[1]
    # py_no_duplicate = sys.argv[2]

    ptr_in = open(py_text, "r")
    # ptr_out = open(py_no_duplicate, "w+")
    map_ = map_read(ptr_in)




