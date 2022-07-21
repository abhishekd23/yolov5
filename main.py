import streamlit as st
import os
import torch
from PIL import Image
import glob
import zipfile
import cv2
import mlflow
import logging
import numpy as np

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def main(dir):
    # st.title('Test Predict YoloV5')
    # wt = []

    # if 'dcount' not in st.session_state:
    #     st.session_state['dcount'] = 0
    # if 'dlist' not in st.session_state:
    #     st.session_state['dlist'] = set()
    # pipeline_name = st.sidebar.selectbox(
        #       'Pipeline',
        #       st.session_state['dlist'],
        #       key='pipeline',
        # )
    # imgs = st.file_uploader("Choose Images")


    # st.markdown("{}".format(pipeline_name))

    # if not imgs:
    #     return
    # else:
    with mlflow.start_run(run_name = "yolov5"):


        for filename in os.listdir(dir):
            img = cv2.imread(os.path.join(dir,filename))
            preds = model(img)
            #print(preds.imgs)
            #preds.show()
            df = preds.pandas().xyxy[0]
            print(df['xmin'])
            mlflow.log_metric("x_min", float(df.xmin))
            mlflow.log_metric("y_min", float(df.ymin))
            mlflow.log_metric("x_max", float(df.xmax))
            mlflow.log_metric("y_max", float(df.ymax))


        #cv2.imshow(preds.imgs,'img')


        # with zipfile.ZipFile(imgs,"r") as zipf:
        #     st.session_state['dcount'] += 1
        #     st.session_state['dlist'].add(st.session_state['dcount'])
        #     zipf.extractall("dataset/v{}".format(st.session_state['dcount']))

        # imgname =os.listdir("dataset/v{}".format(st.session_state['dcount']))
        # preds =glob.glob("dataset/v{}/*.*".format(st.session_state['dcount']),recursive=True)

        # results = model(preds)
        # # results.imgs
        # results.render()
        # os.mkdir("output/v{}".format(st.session_state['dcount']))
        # for index,im in enumerate(results.imgs):

        #     img = Image.fromarray(im)
        #     #dir = 'output/v{}/'.format(st.session_state['dcount'])

        #     st.write(st.session_state['dlist'])
        #     img.save('output/v{}/{}'.format(st.session_state['dcount'], imgname[index]))

        #     st.image('output/v{}/{}'.format(st.session_state['dcount'], imgname[index]))

    # st.button('Predict')

# dcount = 0
# dlist = set()
if __name__ == '__main__':
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s',pretrained=True, _verbose=False)
    model.classes = [0]
    dir = 'images'
    main(dir)