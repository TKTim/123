import sys

car_num = 60
max_map = 10

def cal_pri(num_, ptr_out):
    for j in range(car_num):
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
                print("[{} {:.5f}]".format(num_[j][i], pri_[i]) , end="", file=ptr_out)
            print("", file=ptr_out)
    print("Stop")


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


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('no argument')
        sys.exit()
    py_text = sys.argv[1]
    py_write = sys.argv[2]

ptr_in = open(py_text, "r")
ptr_out = open(py_write, "w+")

lines = (line.rstrip() for line in ptr_in.readlines())  # All lines including the blank ones. Skip first line.
lines = (line for line in lines if line)  # Non-blank lines
Start_load = False
num_ = [[-1 for _ in range(max_map)] for _ in range(car_num)]
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
                print("This is ", iter_, " iterations.", file=ptr_out )
                cal_pri(num_, ptr_out)
                print(num_)
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
print("Finished.")


