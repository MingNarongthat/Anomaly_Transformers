import cv2
import glob
import os

pathname = glob.glob('/Users/mingnarongthat/Documents/Ph.D./Transformer/SwinBERT/datasets/LS/raw_videos/val_all/erM7d0vUWz0_000002_000007.mp4')
pathout = '/Users/mingnarongthat/Documents/Ph.D./Transformer Model/dataset/Image/Training/'

for filename_input in pathname:
    outname = os.path.basename(filename_input)[:-4]
    vidcap = cv2.VideoCapture(filename_input)
    vidcap.set(1, 100)
    success, image = vidcap.read()

    cv2.imwrite("{}frame{}.jpg".format(pathout, outname), image)


