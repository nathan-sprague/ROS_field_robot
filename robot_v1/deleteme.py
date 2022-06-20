#!/usr/bin/python3
import math
n = 21
# Your code should be below this line

if n!=int(n) or n > 31 or n < 1:
	print("Not valid")
elif n%7 == 1 or n%7 == 2:
	print("Weekend")
else:
	print("Weekday")
