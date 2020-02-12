import os
f = open("korean_2350.txt", 'r')

txt = open("ko.txt", 'w')

line  = f.readline()

for cha in line:
    txt.write(cha + '\n')

f.close()
txt.close()