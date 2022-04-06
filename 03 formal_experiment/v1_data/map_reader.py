import sys


def map_read(ptr_in):
    Big_map_max = 0
    Smol_map_max = 0

    lines = (line.rstrip() for line in ptr_in.readlines())  # All lines including the blank ones. Skip first line.
    lines = (line for line in lines if line)  # Non-blank lines

    for line in lines:
        # print(line)
        temp = line.split(' ')
        for i in temp:
            # print(temp)
            temp_in = i.split(',')
            try:
                if int(temp_in[0]) > Smol_map_max:
                    Smol_map_max = int(temp_in[0])
                if int(temp_in[1]) > Big_map_max:
                    Big_map_max = int(temp_in[1])
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

    return Smol_map_max


if __name__ == '__main__':
    py_text = sys.argv[1]

    ptr_in = open(py_text, "r")
    Smol_map_max = map_read(ptr_in)
