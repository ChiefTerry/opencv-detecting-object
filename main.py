import cv2 # Import the OpenCV library
import numpy as np # Import Numpy library
import copy
import time

cap = cv2.VideoCapture(0)
green = (0, 255, 0)
pink = (255, 0, 255)
thickness = 4
MAX_AREA = 300000
MIN_AREA = 40000
min_point_contour = 6
max_point_contour = 10
width_cam  = cap.get(3) # float
height_cam = cap.get(4) # float
min_radius = 200
max_radius = 300
color_tune = (0,0,0)
record_time = 53
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# print("Width cam", width_cam, "Height", height_cam)


def empty():
    pass

cv2.namedWindow('Parameters')
cv2.resizeWindow('Parameters', 640, 240)
cv2.createTrackbar('Threshold1', 'Parameters', 20, 255, empty)
cv2.createTrackbar('Threshold2', 'Parameters', 43, 255, empty)

max_value = 255
max_value_H = 360//2
low_H = 89
low_S = 103
low_V = 29
high_H = 164
high_S = 255
high_V = 179
window_capture_name = 'Video Capture'
window_detection_name = 'Object Detection'
low_H_name = 'Low H'
low_S_name = 'Low S'
low_V_name = 'Low V'
high_H_name = 'High H'
high_S_name = 'High S'
high_V_name = 'High V'

cv2.namedWindow(window_capture_name)
cv2.namedWindow(window_detection_name)
cv2.createTrackbar(low_H_name, window_detection_name , low_H, max_value_H, empty)
cv2.createTrackbar(high_H_name, window_detection_name , high_H, max_value_H, empty)
cv2.createTrackbar(low_S_name, window_detection_name , low_S, max_value, empty)
cv2.createTrackbar(high_S_name, window_detection_name , high_S, max_value, empty)
cv2.createTrackbar(low_V_name, window_detection_name , low_V, max_value, empty)
cv2.createTrackbar(high_V_name, window_detection_name , high_V, max_value, empty)

# def find_circle(img, shown_img, center_of_contour):
#     # Convert to grayscale. 
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # Blur using 3 * 3 kernel.
#     gray_blurred = cv2.blur(gray, (3, 3))

#     img_canny = cv2.Canny(noise_removal, 20, 43)

#     img_dilation = cv2.dilate(img_canny, kernel, iterations = 1)

#     # Apply Hough transform on the blurred image. 
#     detected_circles = cv2.HoughCircles(img_dilation,
#                     cv2.HOUGH_GRADIENT, 1, 20, param1 = 50,
#                 param2 = 30
#                 , minRadius = min_radius, maxRadius = max_radius)

#     # Draw circles that are detected. 
#     if detected_circles is not None:
#         print('Found Circle')

#         # Convert the circle parameters a, b and r to integers. 
#         detected_circles = np.uint16(np.around(detected_circles))

#         for pt in detected_circles[0, :]:
#             a, b, r = pt[0], pt[1], pt[2]
#             # x, y = center_of_contour

#             # if 0 > abs(x - a) > 10 and  0 > abs(y - b) > 10:
#             #     # Draw the circumference of the circle. 
#             #     cv2.circle(shown_img, (a, b), r, (0, 255, 0), 2)

#             #     # Draw a small circle (of radius 1) to show the center. 
#             #     cv2.circle(shown_img, (a, b), 1, (0, 0, 255), 3)
#             # else:
#             # Draw the circumference of the circle. 
#             cv2.circle(shown_img, (a, b), r, (0, 255, 0), 2)

#             # Draw a small circle (of radius 1) to show the center. 
#             cv2.circle(shown_img, (a, b), 1, (0, 0, 255), 3)

#         return True

def detect_circle(img, contour, area):
    (x,y),radius = cv2.minEnclosingCircle(contour)
    center = (int(x),int(y))
    radius = int(radius)
    circle_area = 3.14 * (radius ** 2)

    if 0.9 < circle_area / area < 1.1:
        return True
    else:
        return False
    

def point_prediction(img, approx, area, contour, point, width, height):
    x = point[0]
    y = point[1]
    shape = ""

    # Prediction
    if len(approx) == 8:
        shape = "Sphere or Cylinder"
        if detect_circle(img, contour, area):
            shape = "Circle"
        else:
            shape = "Cylinder"
    elif len(approx) == 6 or len(approx) == 7:
        shape = "Cuboid"
    elif len(approx) > 10:
        shape = "Unknown"

    # print( "Original" , x, y, width, height)
    cv2.putText(img, "Prediction: {}".format(shape), (x + width + 20, y + 70), \
                    cv2.FONT_HERSHEY_COMPLEX, 1.0, (0,0,255), 2)

    return shape

def find_contour(img, imgContour):
    contours, hierarchy = cv2.findContours(imgContour, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[-2:]
    warp = None
    copy_img = copy.copy(img)
    center_of_contour = (0,0)

    if len(contours) > 1:
        con = max(contours, key=cv2.contourArea)
        extLeft = tuple(con[con[:, :, 0].argmin()][0])
        extRight = tuple(con[con[:, :, 0].argmax()][0])
        extTop = tuple(con[con[:, :, 1].argmin()][0])
        extBot = tuple(con[con[:, :, 1].argmax()][0])

        # cv2.putText(img, "left", extLeft, cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 0, 255), 2)
        # cv2.putText(img, "right", extRight, cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 0, 255), 2)
        # cv2.putText(img, "top", extTop, cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 0, 255), 2)
        # cv2.putText(img, "bottom", extBot, cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 0, 255), 2)

        shape = ""

        for contour in contours:
            area = cv2.contourArea(contour)

            if MIN_AREA < area < MAX_AREA:

                # ----------------Draw enclosing circle --------------------------------
                (x,y),radius = cv2.minEnclosingCircle(contour)
                center = (int(x),int(y))
                radius = int(radius)

                circle_area = 3.14 * (radius ** 2)
                print(circle_area / area)
                if 0.9 < circle_area / area < 1.1:
                    cv2.putText(img, "Circle", center, cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 0, 255), 2)
                cv2.circle(img,center,radius,(255,255,0),2)

                # Get perimeter and approximation of boundary
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

                # Apply bounding rectangle
                if min_point_contour <= len(approx) <= max_point_contour:
                    x, y, width, height = cv2.boundingRect(approx)
                    initial_point = (x, y)
                    rec_width = x + width
                    rec_height = y + height
                    pts = np.float32(approx)
                    center_of_contour = (int(x + width/2), int(y + height/2))
                    

                    # cv2.circle(img, center_of_contour, 3, (255, 0, 0), 4)
                    # cv2.putText(img, str(center_of_contour), center_of_contour, \
                    #     cv2.FONT_HERSHEY_COMPLEX, 1.0, (), 2)

                    # Display text
                    cv2.putText(img, "Points: " + str(len(approx)), (x + width + 20, y + 20), \
                        cv2.FONT_HERSHEY_COMPLEX, 1.0, (0,0,255), 2)
                    cv2.putText(img, "Area: " + str(int(area)), (x + width + 20, y + 45), \
                        cv2.FONT_HERSHEY_COMPLEX, 1.0, (0,0,255), 2)

                    width_ROI = int(rec_width * 10 / 9)
                    height_ROI = int(rec_height * 10 / 9)
                    x_ROI = x - int(width_ROI / 10)
                    y_ROI = y - int(height_ROI / 10)
                    center_of_contour = (width_ROI / 2, height_ROI / 2)
                    print("-----------------------------------------------")
                    print("x_ROI        {}".format(x_ROI))
                    print("y_ROI        {}".format(y_ROI))
                    print("width_ROI    {}".format(width_ROI))
                    print("height_ROI   {}".format(height_ROI))
                    print("-----------------------------------------------")

                    # cv2.rectangle(img, (x_ROI, y_ROI), (width_ROI, height_ROI), pink, 7)

                    warp = copy_img[y_ROI: height_ROI, x_ROI: width_ROI, :] \
                        if x_ROI > 0 and y_ROI > 0 \
                        else img[y: y + height, x: x + width, :]

                    print("Img shape", img.shape)
                    print("Warp shape", warp.shape)

                    if 0 in warp.shape:
                        return None

                    shape = point_prediction(img, approx, area, contour, initial_point, width, height)  
                    
                for pts in approx:
                    point = tuple(pts[0])
                    # print(point)
                    cv2.circle(img, point, 3, green, 4)


                for n in range(len(approx)):
                    point = tuple(approx[n][0])
                    cv2.putText(img, str(n + 1), point, cv2.FONT_HERSHEY_COMPLEX, 1.0, (0,0,255), 2)

    return (warp, center_of_contour)

found_contours = []

start_time = time.time()
state = True
counter = 0

while True:
    ret, frame = cap.read()
    # frame = cv2.imread("ping_pong.jpg")

    if state:
        # Gaussian Blur
        img_blur = cv2.GaussianBlur(frame, (7, 7), 1)
        img_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)

        # Get track position
        threshold1 = cv2.getTrackbarPos('Threshold1', 'Parameters')
        threshold2 = cv2.getTrackbarPos('Threshold2', 'Parameters')

        # Remove noise
        noise_removal = cv2.bilateralFilter(img_gray,9,75,75)
        img_canny = cv2.Canny(noise_removal, threshold1, threshold2)

        # gray = cv2.cvtColor(img_blur, cv2.COLOR_GRAY2BGR)
        canny = cv2.cvtColor(img_canny, cv2.COLOR_GRAY2BGR)
        # Convert BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # fourth_frame = np.hstack((img_blur, img_gray, canny))
        # cv2.imshow("Fourth". fourth_frame)
        cv2.imshow("canny", img_canny)
        cv2.imshow("img_blur", img_blur)
        cv2.imshow("gray", img_gray)

        # define range of blue color in HSV
        # lower_blue = np.array([89, 103, 29])
        
        # upper_blue = np.array([164, 255, 179])
        low_H = cv2.getTrackbarPos(low_H_name, window_detection_name)
        low_S = cv2.getTrackbarPos(low_S_name, window_detection_name)
        low_V = cv2.getTrackbarPos(low_V_name, window_detection_name)
        high_H = cv2.getTrackbarPos(high_H_name, window_detection_name)
        high_S = cv2.getTrackbarPos(high_S_name, window_detection_name)
        high_V = cv2.getTrackbarPos(high_V_name, window_detection_name)

        mask = cv2.inRange(hsv, (low_H, low_S, low_V), (high_H, high_S, high_V))

        # Threshold the HSV image to get only blue colorss

        # Bitwise-AND mask and original image
        result = cv2.bitwise_and(frame,frame, mask= mask)

        # cv2.imshow('frame',frame)
        # cv2.imshow('mask',mask)
        cv2.imshow('res',result)

        # Dialation image
        kernel = np.ones((3,3))
        img_dilation = cv2.dilate(img_canny, kernel, iterations = 1)
        img_dilation_to_rgb = cv2.cvtColor(img_dilation, cv2.COLOR_GRAY2BGR)

        warp, center_of_contour = find_contour(frame, img_dilation)

        # if warp is not None:
        #     # found_circle = find_circle(warp, warp, center_of_contour)
        #     cv2.imshow("Warp", warp)
        
        end_time = time.time()

        # print(shape)
        # counter = counter + 1
        # print(counter)

        # if shape != "":
        #     print(":break")
        #     break
        # elif shape == "":
        #     print(end_time - start_time)
        #     start_time = time.time()
            

        # Display the resulting frame
        first_frame = np.concatenate((img_canny, img_dilation), axis= 1)
        second_frame = np.concatenate((frame,img_blur), axis = 1)
        third_frame = np.hstack((frame, img_dilation_to_rgb, result))

        cv2.imshow('Color/Conputer Vision', third_frame)
        
    
        # cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if (cv2.waitKey(1) & 0xFF == ord('w')) and (warp is not None):
        cv2.imwrite('circle.png', warp)
        print('Write successfully')
        cap.release()
        cv2.destroyAllWindows()
        break

    if len(found_contours) == 10 and 0 not in found_contours:
        break

    if cv2.waitKey(1) & 0xFF == ord('x'):
        state = False
        print(state)

    elif cv2.waitKey(1) & 0xFF == ord('x') and state == False:
        state = True
        print(state)

# Close down the video stream
# cap.release()
# cv2.destroyAllWindows()