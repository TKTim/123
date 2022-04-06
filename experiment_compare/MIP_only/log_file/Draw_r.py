import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import sys
import os

'''
Read Reward_File
'''

py_in = sys.argv[1]
py_text = open(py_in, 'r')
ep_r_set = []

lines = (line.rstrip() for line in py_text.readlines())  # All lines including the blank ones. Skip first line.
lines = (line for line in lines if line)  # Non-blank lines

temp = ""
skipping = False
reading = 0
count = 0

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



plt.plot(ep_r_set)
plt.title("EP_r")
plt.grid(True)
plt.tick_params(axis='both',which='major',labelsize=14)
plt.xlim(-10, count + 20)
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
