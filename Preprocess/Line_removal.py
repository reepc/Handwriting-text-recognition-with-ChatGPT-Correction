import numpy as np
import cv2
from scipy.signal import argrelextrema

class Line_Removal:
    def __init__(self, gray):
        self.original = thresh_image(gray)
        self.height, self.width = gray.shape[:2]
    
    def line_distance(self):
        new_width = int(self.width / 5)
        
        lost = int(self.height - int(self.height / new_width) * new_width)
        
        hs_1 = lost + 2 * new_width
        he_1 = lost + 3 * new_width
        hs_2 = lost + 4 * new_width
        he_2 = lost + 5 * new_width
        
        ws_1 = new_width
        we_1 = 2 * new_width
        ws_2 = 3 * new_width
        we_2 = 4 * new_width
        
        window_1 = self.original[hs_1:he_1, ws_1:we_1]
        window_2 = self.original[hs_1:he_1, ws_2:we_2]
        window_3 = self.original[hs_2:he_2, ws_1:we_1]
        window_4 = self.original[hs_2:he_2, ws_2:we_2]
        
        hough_11, hough_12 = self.find_Hough_Lines(window_1)
        hough_21, hough_22 = self.find_Hough_Lines(window_2)
        hough_31, hough_32 = self.find_Hough_Lines(window_3)
        hough_41, hough_42 = self.find_Hough_Lines(window_4)
        
        diff_1 = abs(hough_11[0] - hough_12[0])
        diff_2 = abs(hough_21[0] - hough_22[0])
        diff_3 = abs(hough_31[0] - hough_32[0])
        diff_4 = abs(hough_41[0] - hough_42[0])
        
        diff_arr = np.array([diff_1, diff_2, diff_3, diff_4])
        
        temp = self.height / 17
        result = np.where(diff_arr < temp / 2)
        diff = np.delete(diff_arr, result)
        
        diff[diff > temp * 2.05] = diff[diff > temp * 2.05] / 3
        diff[diff > temp * 1.05] = diff[diff > temp * 1.05] / 2
        
        line_distance = np.median(diff)

        return line_distance
    
    def find_lines(self, distance):
        copy = self.original.copy()
        copy = 255 - copy
        
        original_copy = self.original.copy()

        # invert the image
        original_copy = 255 - original_copy

        # make the image binary
        img_binary = original_copy / 255

        # percentage_width to crop image from the left and right
        percentage_width = 0.25

        # crop image from the left and right end
        img_binary_start = img_binary[:, 0:round(self.width * percentage_width)]
        img_binary_end = img_binary[:, (round(self.width * (1 - percentage_width))-0):(self.width - 200)]
        
        ruling_lines_start = self.find_StartEndPoints(img_binary_start, distance)
        
        # find the start points of the ruling lines from the right
        ruling_lines_end = self.find_StartEndPoints(img_binary_end, distance)
        
        return ruling_lines_start, ruling_lines_end
    
    def draw_ruling_lines(self, ruling_lines, startorend, original_page):
        """ Functions draw the ruling lines
            Parameters:
                ruling_lines    : y-coordinate of the found ruling lines
                startorend      : left or right part of the image
                original_page   : original cropped image
            Returns:
                -
        """

        # creating new images to draw the lines in
        # cdst - ruling lines on top of the original image
        # cdst2 - ruling lines on white background
        # cdst3 - ruling lines on top of the binary image
        cdst = cv2.cvtColor(original_page, cv2.COLOR_GRAY2BGR)
        cdst2 = 255 - cdst

        line_thickness = 1

        for i in range(len(ruling_lines)):
            current = ruling_lines[i]
            pt1 = (0, current)
            pt2 = (np.size(original_page, 1), current)
            cv2.line(cdst, pt1, pt2, (0, 255, 0), line_thickness)  # RGB
            cv2.line(cdst2, pt1, pt2, (0, 255, 0), line_thickness)  # RGB

        cv2.imwrite('test' + "_" + startorend + '_with_text.png', cdst)
        cv2.imwrite('test' + "_" + startorend + '_on_binary_image.png', cdst2)

    
    def find_Hough_Lines(self, window):
        edges = cv2.Canny(window, 50, 150, apertureSize=3)
        
        threshold = (np.size(window, 1) // 3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold)
        
        return np.squeeze(lines[0]), np.squeeze(lines[1])
    
    def find_StartEndPoints(self, img_binary, line_distance):
        # Horizontal Projection
        horizontal_sum = (cv2.reduce(img_binary, 1, cv2.REDUCE_SUM, dtype=cv2.CV_64FC1))

        kernel_size = 51
        # Blurring to smooth the horizontal projection
        signal_smooth = cv2.GaussianBlur(horizontal_sum, (kernel_size, kernel_size), 0, 0)

        # Find the local extremum of the signal
        max_indexes = argrelextrema(signal_smooth, np.greater)
        maxim = max_indexes[0]

        # interval for searching ruling lines between peaks
        search_peak_interval = 40
        lower_bound = int(line_distance) - search_peak_interval
        upper_bound = int(line_distance) + search_peak_interval + 1
        search_peak = np.arange(start=lower_bound, stop=upper_bound)

        # interval for finding the real maximum of the signal - around 60 is the ideal number
        if maxim[0] < 60:
            search_max_interval = maxim[0]
        else:
            search_max_interval = 59

        # a list to store the y-axis position of the ruling lines
        ruling_lines = []

        # setting current to be the first maximum
        i = 0
        current = maxim[0]

        # loop to find the real ruling lines
        while i < len(maxim)-1 and current <= maxim[-1] and len(ruling_lines) < 22:
            mask = np.isin(maxim, current + search_peak)
            if np.any(mask):
                current = current-search_max_interval+np.argmax(horizontal_sum[current-search_max_interval:current+search_max_interval])
                ruling_lines.append(current)
                current = maxim[mask][0]
                k = current
                if k == maxim[-1]:
                    current = current - search_max_interval + np.argmax(horizontal_sum[current - search_max_interval:current+search_max_interval])
                    ruling_lines.append(current)
                    break
            else:
                del ruling_lines[:]
                i += 1
                current = maxim[i]

        return ruling_lines
    
    def create_lineIterator(self, P1, P2, img):
        # define local variables for readability
        imageH = img.shape[0]
        imageW = img.shape[1]
        P1X = P1[0]
        P1Y = P1[1]
        P2X = P2[0]
        P2Y = P2[1]

        # difference and absolute difference between points
        # used to calculate slope and relative location between points
        dX = P2X - P1X
        dY = P2Y - P1Y
        dXa = np.abs(dX)
        dYa = np.abs(dY)

        # predefine numpy array for output based on distance between points
        itbuffer = np.empty(shape=(np.maximum(dYa, dXa), 3), dtype=np.float32)
        itbuffer.fill(np.nan)

        # Obtain coordinates along the line using a form of Bresenham's algorithm
        negY = P1Y > P2Y
        negX = P1X > P2X
        if P1X == P2X:  # vertical line segment
            itbuffer[:, 0] = P1X
            if negY:
                itbuffer[:, 1] = np.arange(P1Y - 1, P1Y - dYa - 1, -1)
            else:
                itbuffer[:, 1] = np.arange(P1Y + 1, P1Y + dYa + 1)
        elif P1Y == P2Y:  # horizontal line segment
            itbuffer[:, 1] = P1Y
            if negX:
                itbuffer[:, 0] = np.arange(P1X - 1, P1X - dXa - 1, -1)
            else:
                itbuffer[:, 0] = np.arange(P1X + 1, P1X + dXa + 1)
        else:  # diagonal line segment
            steepSlope = dYa > dXa
            if steepSlope:
                slope = dX.astype(np.float32) / dY.astype(np.float32)
                if negY:
                    itbuffer[:, 1] = np.arange(P1Y - 1, P1Y - dYa - 1, -1)
                else:
                    itbuffer[:, 1] = np.arange(P1Y + 1, P1Y + dYa + 1)
                itbuffer[:, 0] = (slope * (itbuffer[:, 1] - P1Y)).astype(np.int) + P1X
            else:
                slope = dY.astype(np.float32) / dX.astype(np.float32)
                if negX:
                    itbuffer[:, 0] = np.arange(P1X - 1, P1X - dXa - 1, -1)
                else:
                    itbuffer[:, 0] = np.arange(P1X + 1, P1X + dXa + 1)
                itbuffer[:, 1] = (slope * (itbuffer[:, 0] - P1X)).astype(np.int) + P1Y

        # Remove points outside of image
        colX = itbuffer[:, 0]
        colY = itbuffer[:, 1]
        itbuffer = itbuffer[(colX >= 0) & (colY >= 0) & (colX < imageW) & (colY < imageH)]

        # Get intensities from img ndarray
        itbuffer[:, 2] = img[itbuffer[:, 1].astype(np.uint), itbuffer[:, 0].astype(np.uint)]

        return itbuffer
    
    def line_iterator(
        self,
        ruling_lines_start,
        ruling_lines_end,
        window_start,
        window_end
    ):
        original_page_for_removal = np.copy(self.original)

        if len(ruling_lines_start) == len(ruling_lines_end):
            pass
        elif len(ruling_lines_start) > len(ruling_lines_end):
            diff = len(ruling_lines_start) - len(ruling_lines_end)
        
        # Line Iterator
        for i in range(len(ruling_lines_start) - diff):
            start_point = np.array([0, ruling_lines_start[i]])
            # offset, so that lines with text that extend beyond ruling lines are unaffected by the removal
            offset = 220
            end_point = np.array([self.width - offset, ruling_lines_end[i]])
            end_point = np.array([self.width, ruling_lines_end[i]])
            points = self.create_lineIterator(start_point, end_point, original_page_for_removal)
            for j in range(len(points)):
                x = int(points[j][0])
                y = int(points[j][1])
                vertical = original_page_for_removal[y - window_start: y + window_end, x]
                if np.count_nonzero(vertical == 0) <= 5:
                    vertical[vertical == 0] = 255

        filepath_rl_removed_page = './removal_test.jpg'
        # image without the ruling lines is written to a file
        cv2.imwrite(filepath_rl_removed_page, original_page_for_removal)

        # a binary image containing the pixels removed by the chosen algorithm is calculated
        removed_pixels = original_page_for_removal - self.original

        # uncomment below to write a binary image containing all the removed_pixels onto the vis directory
        # filepath_rl_removed_px = cnf.folder_vis + self.filename + "_lineIterator_removed_pixels" + ".png"
        # cv2.imwrite(filepath_rl_removed_px, removed_pixels)

        return removed_pixels

def thresh_image(unshadowed):
    threshold = np.max(unshadowed) - np.std(unshadowed)
    binary = np.where(unshadowed > threshold, 255, 0).astype(np.uint8)
    
    return binary