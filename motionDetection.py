import cv2
import numpy as np

camera = cv2.VideoCapture(0)
background = None
fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    ret,frame = camera.read()

    if not ret:
        break
    fgmask = fgbg.apply(frame)

    # Color doesn't matter in motion detection, so convert to grayscale
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # Blur image to account for inconsistencies between frames, no two frames will be the same
    # due to sensors variations (some pixels will have intensity changes)
    blur = cv2.GaussianBlur(gray,(7,7),0)

    # let's grab the first frame of the camera if it hasn't been initialized already
    # we are assuming the first frame will only contain the background and no moving objects
    if background is None:
        background = gray
        continue

    # Now we need to perform background subtraction
    diff = cv2.absdiff(background,gray)
    #thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.threshold(fgmask,25,255,cv2.THRESH_BINARY)[1]


    #cv2.imshow("foreground", fgmask)
    #cv2.imshow("diff", diff)
    #cv2.imshow("thresh",thresh)

    # dilate the thresholded image to fill in holes, then find contours
    # on thresholded image
    thresh = cv2.dilate(thresh, None, iterations=2)
    (img,cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    # loop over the contours
    for c in cnts:
        if cv2.contourArea(c) < 5000:
            continue
        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow("frame", frame)
    # if the 'q' key is pressed, stop the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# release the video capture
camera.release()
cv2.destroyAllWindows()

