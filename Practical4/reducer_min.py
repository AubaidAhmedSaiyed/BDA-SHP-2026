#!/usr/bin/env python3
import sys

current_year = None
min_temp = None

for line in sys.stdin:
    year, temp = line.strip().split("\t")
    temp = int(temp)

    if year != current_year:
        if current_year is not None:
            print(f"{current_year}\t{min_temp}")
        current_year = year
        min_temp = temp
    else:
        min_temp = min(min_temp, temp)

if current_year is not None:
    print(f"{current_year}\t{min_temp}")
    