#!/usr/bin/env python3
import sys

for line in sys.stdin:
    line = line.strip()
    if len(line) < 93:
        continue
    year = line[15:19]
    if line[87] == '+':
        temp = int(line[88:92])
    else:
        temp = int(line[87:92])
    quality = line[92]
    if temp != 9999 and quality in "01459":
        print(f"{year}\t{temp}")