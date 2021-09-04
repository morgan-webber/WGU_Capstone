import random
import glob
import cv2
import numpy as np
from PyQt5.QtWidgets import QLabel

class ChadAI:

    mutation_chance = 0.1
    CLASS_LIGHT = 1
    CLASS_MEDIUM = 2
    CLASS_HEAVY = 3

    light_label_style = "QLabel { background-color: black; color: green;}"
    medium_label_style = "QLabel { background-color: black; color: yellow;}"
    heavy_label_style = "QLabel { background-color: black; color: red;}"

    current_video = None
    report_label = None

    def __init__(self):
        self.light = []
        self.medium = []
        self.heavy = []
        self.bigLight = []
        self.smallMedium = []
        self.bigMedium = []
        self.smallHeavy = []

        # Boundaries for light/medium/heavy
        self.light_medium_boundary = 59
        self.medium_heavy_boundary = 83

        # Generation ID
        self.generationID = 0

    def addLight(self, data):
        self.light.append(data)

    def addMedium(self, data):
        self.medium.append(data)

    def addHeavy(self, data):
        self.heavy.append(data)

    def addBigLight(self, data):
        self.bigLight.append(data)

    def addSmallMedium(self, data):
        self.smallMedium.append(data)

    def addBigMedium(self, data):
        self.bigMedium.append(data)

    def addSmallHeavy(self, data):
        self.smallHeavy.append(data)

    def getMinArray(self, arr):
        try:
            return round(min(arr, key=lambda x:x[1])[1], 2)
        except ValueError:
            return 0

    def getMaxArray(self, arr):
        try:
            return round(max(arr, key=lambda x:x[1])[1], 2)
        except ValueError:
            return 0

    def getMinLight(self):
        return self.getMinArray(self.light)

    def getMaxLight(self):
        return self.getMaxArray(self.light)

    def getMinMedium(self):
        return self.getMinArray(self.medium)

    def getMaxMedium(self):
        return self.getMaxArray(self.medium)

    def getMinHeavy(self):
        return self.getMinArray(self.heavy)

    def getMaxHeavy(self):
        return self.getMaxArray(self.heavy)

    def printRangeDefinition(self):
        print("Light: %s - %s   Medium: %s - %s   Heavy: %s - %s" % (self.getMinLight(), self.getMaxLight(),
                                                                     self.getMinMedium(), self.getMaxMedium(),
                                                                     self.getMinHeavy(), self.getMaxHeavy()))

    def processVideo(self):
        cap = cv2.VideoCapture(self.current_video)
        detector = cv2.createBackgroundSubtractorMOG2(128, cv2.THRESH_BINARY, 0)

        # Begin processing
        final_class = self.CLASS_LIGHT
        while True:
            ret, frame = cap.read()

            if frame is None:
                break

            # Define region of interest
            height, width, _ = frame.shape

            # Define polygon region of interest
            pts = [(213, 236), (179, 168), (167, 119), (280, 119), (319, 148), (320, 240)]
            points = np.array(pts, np.int32)
            points = points.reshape((-1, 1, 2))
            polyROI = np.zeros(frame.shape, np.uint8)
            polyROI = cv2.polylines(polyROI, [points], True, (255, 255, 255), 2)
            polyROI2 = cv2.fillPoly(polyROI.copy(), [points], (255, 255, 255))
            polyROIfinal = cv2.bitwise_and(polyROI2, frame)

            # Canny edge detection
            edges = cv2.Canny(polyROIfinal, 150, 200)

            # Object detection
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Classify
            if len(contours) <= self.light_medium_boundary:
                self.setClassLabel(self.CLASS_LIGHT, self.report_label)
                final_class = self.CLASS_LIGHT
            elif len(contours) > self.light_medium_boundary and len(contours) <= self.medium_heavy_boundary:
                self.setClassLabel(self.CLASS_MEDIUM, self.report_label)
                final_class = self.CLASS_MEDIUM
            elif len(contours) > self.medium_heavy_boundary:
                self.setClassLabel(self.CLASS_HEAVY, self.report_label)
                final_class = self.CLASS_HEAVY

            key = cv2.waitKey(30)
            if key == 27:
                break

    def setClassLabel(self, rank, label: QLabel):
        if rank == ChadAI.CLASS_LIGHT:
            label.setText("Light")
            label.setStyleSheet(self.light_label_style)
        elif rank == ChadAI.CLASS_MEDIUM:
            label.setText("Medium")
            label.setStyleSheet(self.medium_label_style)
        elif rank == ChadAI.CLASS_HEAVY:
            label.setText("Heavy")
            label.setStyleSheet(self.heavy_label_style)

    def train(self, videoFiles, info):
        for file in videoFiles:

            cap = cv2.VideoCapture(file)
            # detector = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=2000, detectShadows=False)
            detector = cv2.createBackgroundSubtractorMOG2(128, cv2.THRESH_BINARY, 0)

            numContours = []

            while True:
                ret, frame = cap.read()

                if frame is None:
                    break

                # Convert to greyscale
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Dilate
                # frame = cv2.dilate(frame, np.ones((1, 1), np.uint8))

                # Define region of interest
                height, width, _ = frame.shape
                roi = frame[90: height - 50, 180:width - 70]

                # Define polygon region of interest
                pts = [(213, 236), (179, 168), (167, 119), (280, 119), (319, 148), (320, 240)]
                points = np.array(pts, np.int32)
                points = points.reshape((-1, 1, 2))
                polyROI = np.zeros(frame.shape, np.uint8)
                polyROI = cv2.polylines(polyROI, [points], True, (255, 255, 255), 2)
                polyROI2 = cv2.fillPoly(polyROI.copy(), [points], (255, 255, 255))
                polyROI3 = cv2.fillPoly(polyROI.copy(), [points], (0, 255, 0))
                polyROIfinal = cv2.bitwise_and(polyROI2, frame)
                # cv2.imshow("PolyROI", polyROIfinal)

                # cv2.rectangle(frame, (130, 90), (width - 1, height - 1), (255, 0, 0), 1)

                # Canny edge detection
                mask = detector.apply(polyROIfinal)
                edges = cv2.Canny(polyROIfinal, 150, 200)

                # cv2.imshow("Canny", edges)

                # Object detection

                # _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Grab number of contours in this frame
                numContours.append(len(contours))

                # Contour drawing
                for contour in contours:
                    # Calc area and filter out small ones
                    contour_area = cv2.contourArea(contour)
                    if contour_area > 10:
                        # Grab rectangle
                        x, y, w, h = cv2.boundingRect(contour)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0))
                        # cv2.drawContours(roi, [contour], -1, (0, 255, 0), 1)

                cv2.imshow("Frame", frame)
                # cv2.imshow("Mask", mask)

                key = cv2.waitKey(30)
                if key == 27:
                    break

            # Debugging print to console
            total = 0
            nonZero = 0
            for num in numContours:
                if num != 0:
                    total += num
                    nonZero += 1

            avg = float(total) / nonZero
            filename = file.split("\\")[2]

            # Metric counting
            if info[filename] == "light":
                self.addLight((filename, avg))
            elif info[filename] == "medium":
                self.addMedium((filename, avg))
            elif info[filename] == "heavy":
                self.addHeavy((filename, avg))

            cap.release()

        # Print ranges
        self.printRangeDefinition()
        print("Boundaries: " + str(self.light_medium_boundary) + " " + str(self.medium_heavy_boundary) + "\n")

    def nextGeneration(self):
            child = ChadAI()

            # Determine boundaries for this generation
            upperLight = self.getMaxLight()
            lowerMedium = self.getMinMedium()
            upperMedium = self.getMaxMedium()
            lowerHeavy = self.getMinHeavy()

            light_medium_boundary = lowerMedium - ((lowerMedium - upperLight) / 2.0)
            medium_heavy_boundary = lowerHeavy - ((lowerHeavy - upperMedium) / 2.0)

            light_medium_diff = self.light_medium_boundary - light_medium_boundary
            medium_heavy_diff = self.medium_heavy_boundary - medium_heavy_boundary
            light_medium_power = abs(light_medium_diff) / (10 + abs(light_medium_diff))
            medium_heavy_power = abs(medium_heavy_diff) / (10 + abs(medium_heavy_diff))

            if light_medium_diff > 0: # child is lower than me
                child.light_medium_boundary = self.light_medium_boundary - (abs(light_medium_diff) * light_medium_power)
            else:
                child.light_medium_boundary = self.light_medium_boundary + (abs(light_medium_diff) * light_medium_power)

            if medium_heavy_diff > 0: # child is lower than me
                child.medium_heavy_boundary = self.medium_heavy_boundary - (abs(medium_heavy_diff) * medium_heavy_power)
            else:
                child.medium_heavy_boundary = self.medium_heavy_boundary + (abs(medium_heavy_diff) * medium_heavy_power)

            # Increment generation ID
            child.generationID = self.generationID + 1

            return child
