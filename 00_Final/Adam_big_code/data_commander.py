import sys
import map_reader

# Parameters
max_map = 10
car_num_max = 10
iter_ = 10

def get_pri(count, for_p_temp):
    pri_ = [0.0] * (for_p_temp+1)
    pri_x = 1 / count
    right = 0
    for i in reversed(range(for_p_temp)):
        temp_i = i + 1
        # print("f:", temp_i, "p:", pri_x)
        pri_[right] = temp_i * pri_x
        right += 1

    return pri_

def cal_pri(num_, D3_value_car):  # D3_value_car
    for j in range(car_num_max):
        count = 0
        for_p_temp = 0
        for i in num_[j]:
            if i == -1:
                break
            else:
                for_p_temp += 1
                count += for_p_temp
        if count == 0:
            break
        else:
            pri_ = get_pri(count, for_p_temp)
            for i in range(for_p_temp):
                # print("[{} {:.5f}]".format(num_[j][i], pri_[i]), end="", file=ptr_out)
                D3_value_car[j][i][0] = num_[j][i]
                D3_value_car[j][i][1] = pri_[i]

            # print("", file=ptr_out)

D3_value = [[[[-1, -1] for k in range(max_map)] for j in range(car_num_max)] for i in range(iter_)]

# num_priority
if len(sys.argv) < 2:
    print('no argument')
    sys.exit()
py_text = sys.argv[1]  # map_collect.txt
py_map = sys.argv[2]  # map.txt
py_write = sys.argv[3]  # vec_num.txt

ptr_in = open(py_text, "r")
ptr_map = open(py_map, "r")
ptr_out = open(py_write, "w+")


lines = (line.rstrip() for line in ptr_in.readlines())  # All lines including the blank ones. Skip first line.
lines = (line for line in lines if line)  # Non-blank lines
Start_load = False
num_ = [[-1 for _ in range(max_map)] for _ in range(car_num_max)]
car_num = 0
iter_ = 0
for line in lines:
    pos = 0
    temp = ""
    # print(line)
    for i in line:
        if Start_load:
            if i == "E":  # End of a iter
                Start_load = False
                car_num = 0
                # print("This is ", iter_, " iterations.", file=ptr_out )
                cal_pri(num_, D3_value[iter_])
                # print(num_)
                num_ = [[-1 for _ in range(10)] for _ in range(car_num_max)]
                iter_ += 1
                break
            elif i == " ":
                num_[car_num][pos] = int(temp)
                pos += 1
                temp = ""
            elif i == "G":  # end of line
                temp = ""
                pos = 0
            else:  # save the map number
                temp += i
        elif i == "S":
            Start_load = True
            car_num = -1
            pos = 0
            break
        else:
            continue
    car_num += 1
    pos = 0
print("num_priority Finished.")

# map_reader
small_map_number = map_reader.map_read(ptr_map)

# vec_num_generator

num_list = [[0 for _ in range(small_map_number)] for _ in range(iter_)]  # numbers of maps

# D3_value = [[[[-1, -1] for k in range(max_map)] for j in range(car_num_max)] for i in range(iter_)]

for i in range(iter_):
    for j in range(car_num_max):
        for z in range(max_map):
            index = D3_value[i][j][z][0]
            if index != -1:
                num_list[i][index] += D3_value[i][j][z][1]  # priority value
                print("D3_value[{}][{}][{}]: ".format(i,j,z), D3_value[i][j][z][1])

for i in range(iter_):
    print(" iter: ", i, file=ptr_out)
    print("[", end="", file=ptr_out)
    for j in range(small_map_number):

        print("{:.5f}".format(num_list[i][j]), end=" ", file=ptr_out)
    print("]", file=ptr_out)




