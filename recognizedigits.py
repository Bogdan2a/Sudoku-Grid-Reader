import cv as cv
import pytesseract
from PIL import Image
import glob
import os

image_path = './Squares/square*.bmp'

# sort file names numerically not alphabetically
def numerical_sort(value):
    import re
    numbers = re.findall(r'\d+', value)
    return int(numbers[0]) if numbers else 0

squares = []
filenames = sorted(glob.glob(image_path), key=numerical_sort) 
for filename in filenames:
    im = Image.open(filename)
    squares.append(im)

# Open the output file
with open('founddigits.txt', 'w') as file:
    print(len(squares))
    print("entered in the open part")
    for i, square in enumerate(squares):
        
        # use Tesseract to recognize the digit
        digit = pytesseract.image_to_string(square, config='--psm 10 -c tessedit_char_whitelist=0123456789').strip()

        if digit.isdigit():
            print(f"Digit in cell {i}: {digit}")
            file.write(f"{digit}\n")
        else:
            print(f"No digit found in cell {i}")
            file.write(f"0\n")

print("All digits have been processed and written to founddigits.txt")