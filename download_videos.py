import os
os.system("brew install youtube-dl")
os.system("youtube-dl --batch-file urls.txt --format mp4")