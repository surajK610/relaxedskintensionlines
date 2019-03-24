import line_parts as rs
import cv2
import argparse
import sys

def start_prompt():
    image_loc = raw_input("Please enter the image location:")  # Python 2
    im = cv2.imread(image_loc)

    orientation = raw_input("Please enter the orientation of the picture (straight frontal: sf, semi side: ss): ")

    if "sf" in orientation:
        im = rs.draw_lines_frontal(im)
    elif "ss" in orientation:
        im = rs.draw_lines_semifrontal(im)
    return im, image_loc


try:
    image_loc = sys.argv[1]
    im = cv2.imread(image_loc)

    orientation = sys.argv[2]
    if "sf" in orientation:
        im = rs.draw_lines_frontal(im)
    elif "ss" in orientation:
        im = rs.draw_lines_semifrontal(im)

    output = image_loc.replace(".", "_processed.")
    cv2.imwrite(output, im)
except:
    print "usage: python run_lines.py image_loc orientation(sf, ss, ps)"

