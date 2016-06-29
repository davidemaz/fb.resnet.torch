import sys

import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas as pd
import csv

#The first argument of this script is the logfile to plot
file = sys.argv[1]

version = 2 #default value
if len(sys.argv) > 2:
	version = sys.argv[2]

#Set columns indices depending on the version
if version == 1:
	top1traincol = 10
	top5traincol = 12
	top1testcol = 8
	top5testcol = 12
elif version == 2:
	top1traincol = 11
	top5traincol = 15
	top1testcol = 11
	top5testcol = 15

# Split Train and Test log lines, throw other lines
reader = csv.reader(open(file, 'rb'))
lines = list(reader)

trainrows = []
for row in lines:
    if len(row) > 0 and "| Epoch: [" in row[0]:
        trainrows.append(row[0])
fo = open("tmp_train.txt", "w")
fo.writelines(line + '\n' for line in trainrows)

testrows = []
for row in lines:
    if len(row) > 0 and "| Test: [" in row[0]:
        testrows.append(row[0])
fo = open("tmp_test.txt", "w")
fo.writelines(line + '\n' for line in testrows)

traindata = pd.read_csv("tmp_train.txt", sep='\s+', skiprows=3, error_bad_lines=False)
top1errortrain = traindata.ix[:,top1traincol].as_matrix()
top5errortrain = traindata.ix[:,top5traincol].as_matrix()

testdata = pd.read_csv("tmp_test.txt", sep='\s+', skiprows=3, error_bad_lines=False)
top1errortest = testdata.ix[:,top1testcol].as_matrix()
top5errortest = testdata.ix[:,top5testcol].as_matrix()

#Plot
fig1 = plt.figure(num=None, figsize=(15, 4), dpi=80, facecolor='w', edgecolor='k')
ax1 = fig1.add_subplot(1,2,1)
ax1.plot(top1errortrain, label='top 1')
#ax1.plot(top5errortrain, label='top 5')
ax1.grid()
ax1.legend()
ax1.set_title('Train Error %')
ax2 = fig1.add_subplot(1,2,2)
ax2.plot(top1errortest, label='top 1')
#ax2.plot(top5errortest, label='top 5')
ax2.grid()
ax2.legend()
ax2.set_title('Test Error %')
plt.show()