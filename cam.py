import cv2
import timeit
import numpy as np
import features


def main():
    video_src = -1
    cam = cv2.VideoCapture(video_src)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # get train features
    img = cv2.imread('camalin.png')
    train_features = features.getFeatures(img)
    cur_time = timeit.default_timer()
    frame_number = 0
    scan_fps = 0
    result = cv2.VideoWriter('filename.avi',  
                         cv2.VideoWriter_fourcc(*'MJPG'), 
                         10, (640, 480)) 

    while True:
        frame_got, frame = cam.read()
        h, w = frame.shape[:2]
        if frame_got is False:
            break

        frame_number += 1
        if not frame_number % 100:
            scan_fps = 1 / ((timeit.default_timer() - cur_time) / 100)
            cur_time = timeit.default_timer()

        region = features.detectFeatures(frame, train_features)  # output coordinate in the camera input and distance between the logo and the detected image

        text = scan_fps

        font = cv2.FONT_HERSHEY_COMPLEX_SMALL

        cv2.putText(frame,  
                str(('FPS= {:.2f}'.format(text))),  
                (0, 50),  
                font, 1,  
                (0, 255, 255),  
                2,  
                cv2.LINE_4) 

        if region is not None:
            box = cv2.boxPoints(region)
            box = np.int0(box)
            cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)

        cv2.imshow("Preview", frame)
        result.write(frame)
        if cv2.waitKey(10) == 27:
            break

    frame.release() 
    result.release() 

    print("The video was successfully saved")  

if __name__ == '__main__':
    main()
