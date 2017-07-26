#!/usr/bin/env python3

new_label = "labels.txt"
imagenet_map = "synset_words.txt"
label2num = "label2num.txt"

num_dict = {}

out_file = open(label2num, 'w')

with open(new_label, 'r') as f:
    for line in f:
        if line:
            num = int(line.split(':')[0])
            lab = line.split(':')[1].strip()
            num_dict[lab] = num

print(len(num_dict))

with open(imagenet_map, 'r') as f:
    for line in f:
        if line:
            nn = line[:9]
            lab = line[10:].strip()
            print(nn, lab)
            num = num_dict[lab]
            out_file.write("{}:{}:{}\n".format(num, nn, lab))

