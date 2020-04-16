import numpy as np
import cv2
from tensorflow.keras.models import load_model

import helper


if __name__ == "__main__":
    model_name = 'attractiveNet_mnv2'
    model_path = 'models/' + model_name + '.h5'

    model = load_model(model_path)
    cap = cv2.VideoCapture(0)

    while(True):
        ret, frame = cap.read()
        score = model.predict(np.expand_dims(helper.preprocess_image(frame,(350,350)), axis=0))
        text1 = f'AttractiveNet Score: {str(round(score[0][0],1))}'
        text2 = "press 'Q' to exit"
        cv2.putText(frame,text1, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2, cv2.LINE_AA)
        cv2.putText(frame,text2, (10,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2, cv2.LINE_AA)
        cv2.imshow('AttractiveNet',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()