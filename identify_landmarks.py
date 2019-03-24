from line_parts import draw_lines_semifrontal, draw_lines_frontal
import glob

import cv2


# def interpolate_pts_x(pts):
#     return pts
#
# def interpolate_pts_y(pts):
#     return pts

if __name__ == "__main__":
    DIR = '/home/pikachu/PycharmProjects/langerlines/Scraped_Images/Faces/'#/train/'
    batch = 32
    images= []
    fnames = glob.glob(DIR + '*.jpg')


    for fname in fnames:
        print fname
        #fname_pre = fname.replace('train', 'train_pre')
        #fname_deid = fname.replace('train', 'train_deid').replace('.','_deid.')
        fname_box = fname.replace('Faces', 'Faces_Boxed')

        #im = Image.open(fname)
        #im = face_recognition.load_image_file(fname)
        im = cv2.imread(fname)
        if '3' in fname or '4' in fname or '5' in fname or '6' in fname or '7' in fname or '8' in fname:
        #if '4' in fname:
            im = draw_lines_semifrontal(im)
        else:
            im = draw_lines_frontal(im)

        cv2.imwrite(fname_box, im)

            #im.paste(ic, box)



