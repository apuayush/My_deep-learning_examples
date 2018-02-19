import cv2
from darkflow.net.build import TFNet
import numpy as np
import time

option = {
    'model': 'cfg/yolo.cfg',
    'load': '/home/apurvnit/Projects/darkflow/bin/yolo.weights',
    'threshold': 0.3,
    'gpu': 1.0
}

tfnet = TFNet(option)

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH,1920)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)

colors = [tuple(255 * np.random.rand(3)) for i in range(10)]
c=0
while(1):
    c+=1
    if c%3!=0:
        pass

    stime = time.time()
    ret, frame = cam.read()

    results = tfnet.return_predict(frame)

    if ret:
        for color, result in zip(colors, results):
            tl = (result['topleft']['x'], result['topleft']['y'])
            br = (result['bottomright']['x'], result['bottomright']['y'])

            label = result['label']
            confidence = result['confidence']
            text = '{}: {:0f}%'.format(label, confidence * 100)
            frame = cv2.rectangle(frame, tl, br, color, 5)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.putText(frame, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 2)
        cv2.imshow('frame', frame)
        print('FPS {:.0f}%'.format(1/(time.time()-stime)))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        cam.release()
        cv2.destroyAllWindows()
        break
