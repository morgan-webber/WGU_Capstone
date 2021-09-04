import csv, os

goodvideos = os.listdir("CHAD\TrafficDataset\mp4Video")
avivideos = os.listdir("CHAD\TrafficDataset\\video")

for avi in avivideos:

    # Find the video file
    sel = None
    avi_to_mp4 = avi[:len(avi) - 4] + ".mp4"
    isGood = False
    for video in goodvideos:
        if video == avi_to_mp4:
            isGood = True

    if not isGood:
        fileStr = "CHAD\TrafficDataset\\video\\"
        os.remove(fileStr + avi)
        print("Removed " + avi)
    