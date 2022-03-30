import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import sys
import os

'''
Read Reward_File
'''

py_text = ["2022-03-24-22-33.log"]
ep_r_set = []
for i in py_text:
    f = open(i, 'r')
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
plt.tick_params(axis='both',which='major',labelsize=14)
plt.xlim(-10, 50)
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
