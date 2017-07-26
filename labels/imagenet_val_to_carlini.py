#!/usr/bin/env python3

import os
import sys
import glob

f = open('label2num.txt')

mapping = {}
for line in f:
    l = line.strip().split(':')
    mapping[l[1]] = l[0]

print("Total {} classes loaded".format(len(mapping)))

os.system("mkdir -p imgs")

file_list = glob.glob('val/**/*.JPEG', recursive=True)
print("Total {} files found".format(len(file_list)))

cur = 1
total = len(file_list)

for img_path in file_list:
    s = img_path.split('/')[1]
    n = os.path.splitext(os.path.basename(img_path))[0].split('_')[2]
    label = mapping[s]
    os.system("cp {} imgs/{}.{}.jpg".format(img_path, label, n))
    if cur % 1000 == 0:
        print("{}/{} finished    ".format(cur, total))
    cur += 1

print()

