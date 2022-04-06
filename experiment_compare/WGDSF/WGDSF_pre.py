import sys
import operator as op


def cal_popularity(num_, output):
    for j in range(car_total_num):
        for i in num_[j]:  # i = index of map
            if i != -1:
                output[iter_][i] += 1
    for i in range(map_num):
        output[iter_][i] = output[iter_][i] / require_total_map


def getReward(sorted_H, WTF_separated, vec_num):
    reward_temp = 0
    r_ = 0
    for j in range(B_size, map_num):  # It's a sort array, after the B_size will be the map does not cache.
        if WTF_separated[sorted_H[j][0]] > 0.0:  # Not cache but requested.
            r_ += REWARD_MINUS * vec_num[sorted_H[j][0]]

    return r_


map_num = 2161
iter_num = 200
car_total_num = 60
require_total_map = 10
B_size = 1000
REWARD_MINUS = -15.0

if len(sys.argv) < 2:
    print('no argument')
    sys.exit()
py_text = sys.argv[1]
py_vec = sys.argv[2]

ptr_in = open(py_text, "r")
py_vec_text = open(py_vec, "r")

lines = (line.rstrip() for line in py_vec_text.readlines())  # All lines including the blank ones. Skip first line.
lines = (line for line in lines if line)  # Non-blank lines

vec_num = [[0.0 for _ in range(map_num)] for _ in range(iter_num)]  # numbers of maps
# Read file
iter_time = -1
for line in lines:
    search_ = False
    map_pos = 0
    temp = ""
    for i in line:
        if i == "i":  # ignore iter line
            iter_time += 1
            break
        if search_:  # Starting to search everything
            if i == "]":  # End of the line
                temp = ""
                search_ = False
            elif i == " ":  # pri of one map is collected
                vec_num[iter_time][map_pos] = float(temp)
                map_pos += 1
                temp = ""
            else:  # collecting pri of map
                temp += i
        if i == "[":
            search_ = True

py_vec_text.close()

lines = (line.rstrip() for line in ptr_in.readlines())  # All lines including the blank ones. Skip first line.
lines = (line for line in lines if line)  # Non-blank lines
Start_load = False
num_ = [[-1 for _ in range(require_total_map)] for _ in range(car_total_num)]
WTF_separated = [[0.0 for _ in range(map_num)] for _ in range(iter_num)]
H_ = [[0, 0.0] for _ in range(map_num)]  # [Number, value], Since we will use sort so, no need to confuse it.
# Initial H_, give[0] the index of map.
for i in range(map_num):
    H_[i][0] = i

# print(H_)


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
                cal_popularity(num_, WTF_separated)
                num_ = [[-1 for _ in range(10)] for _ in range(60)]
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
# print(WTF_separated)
min_H = 0.0
WDT = 1
SC = 1.25
WTF_accumulate = [0.0 for _ in range(map_num)]
sorted_H = [[0, 0.0] for _ in range(map_num)]
ep_r = 0
r = 0
for i in range(0, iter_num):
    if i != 0:
        sorted_H = sorted(H_, key = lambda x:x[1])
        # print(sorted_H)
        r = getReward(sorted_H, WTF_separated[i], vec_num[i])
        print("r: ", r)
        ep_r += r

    min_H = sorted_H[0][1]
    for j in range(map_num):
        WTF_accumulate[j] += WTF_separated[i][j]
        H_[j][1] = min_H + SC * WDT * WTF_accumulate[j]

print(ep_r)





