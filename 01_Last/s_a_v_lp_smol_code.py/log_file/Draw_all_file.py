import matplotlib.pyplot as plt
import os
import glob
import platform


'''
Read Reward_File
'''
my_os = platform.system()
if my_os == "Linux":
    glob_temp = './*.log'
else:
    glob_temp = '.\*.log'

ep_r_set = []
count = 0

list_ = sorted(glob.glob(glob_temp))
for filename in list_:
    with open(os.path.join(os.getcwd(), filename), 'r') as f:
        print(filename)
        lines = (line.rstrip() for line in f.readlines())  # All lines including the blank ones. Skip first line.
        lines = (line for line in lines if line)  # Non-blank lines

        temp = ""
        skipping = False
        reading = 0

        # Print every ep_r
        for line in lines:
            skip = 2
            temp = ""
            for i in line:
                if skip == 0:
                    skipping = False
                    if i.isalpha():
                        break
                    elif i == ".":
                        ep_r_set.append(int(temp))
                        count += 1
                        break
                    else:
                        temp += i
                elif skipping:
                    skip -= 1
                elif i == "r":
                    skipping = True

        f.close()

plt.plot(ep_r_set)
plt.title("EP_r")
plt.grid(True)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.xlim(-10, count+20)
plt.show()

# Ever five round
'''
for line in lines:
    skip = 2
    temp = ""
    if reading % 5 == 0:
        for i in line:
            if skip == 0:
                skipping = False
                if i.isalpha():
                    break
                elif i == ".":
                    ep_r_set.append(int(temp))
                    break
                else:
                    temp += i
            elif skipping:
                skip -= 1
            elif i == "r":
                skipping = True
    reading += 1
    '''
