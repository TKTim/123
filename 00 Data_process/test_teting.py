import sys

testing_set_num = 3

dir_ = sys.argv[1]

map_file = dir_ + "map.txt"
ptr_in = open(map_file, "r")
print(ptr_in)

for i in range(1, testing_set_num):
    vec_file = dir_ + "d" + str(i) + "_vec_num.txt"

    f = open(vec_file, 'r')

    print(f)
