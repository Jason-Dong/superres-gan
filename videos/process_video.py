import cv2 
import os
from os.path import isfile, join
import numpy as np

# Function to extract frames 
def convert_video_to_frames(path, func): 
    ''' Extracts frames from video and applies function to each frame
    path - PATH to folder containing video
    func - function to be applied to each frame

    returns new Video with function applied to each frame'''

    # Path to video file 
    vidObj = cv2.VideoCapture(path) 
    count = 0
  
    # checks whether frames were extracted 
    success = 1

    directory = './frames/'
    os.chdir(directory) 
  
    while success: 
        # vidObj object calls read 
        # function extract frames 
        success, image = vidObj.read() 

        #applied function to image
        image = func(image)

        # Saves the frames with frame-count 
        cv2.imwrite("frame%d.jpg" % count, image) 
  
        count += 1

def convert_frames_to_video(pathIn, pathOut, fps):
    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
 
    #for sorting the file names properly
    files.sort(key = lambda x: int(x[5:-4]))
 
    for i in range(len(files)):
        filename=pathIn + files[i]
        #reading each files
        img = cv2.imread(filename)
        try:
        	height, width, layers = img.shape
        	size = (width,height)
        	#print(filename)
        	#inserting the frames into an image array
        	frame_array.append(img)
        except:
        	continue
 
    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'MP4V'), fps, size)
 
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()

if __name__ == '__main__':
	print('[INFO] Processing Video Frames')
	convert_video_to_frames('./bread.mp4', lambda x: x)
	print('[INFO] Compiling Frames')
	pathIn= './frames/'
	pathOut = 'video.mp4'
	fps = 25.0
	convert_frames_to_video(pathIn, pathOut, fps)
