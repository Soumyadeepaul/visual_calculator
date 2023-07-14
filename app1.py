import streamlit as st
import cv2
import numpy as np
import tempfile
import matplotlib.pyplot as plt
import random
import os
import matplotlib.image as mpimg
import math
from tensorflow.keras.models import load_model
from cvzone.HandTrackingModule import HandDetector
import time

#Main function
def main():
    #page title
    st.set_page_config(page_title="Visual Calculator")
    st.markdown("<h1 style='text-align: center; color: black; background-color: white;'>Visual Calculator</h1>",unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center; color: white; background-color: purple; padding-top: 20px;'>Based on: Convolutional Netural Network</h5>",unsafe_allow_html=True)
    #load the model
    model = load_model(r'C:\Users\Lenovo\nn_calculator_project\calculator4.h5')

    # Image resizing to aspect ration 1:1
    def resize_img(img, imgh, imgw):
        imgsize = 80
        imgwhite = np.ones((imgsize, imgsize, 3), np.uint8) * 255

        if imgh > imgw:
            k = imgsize / imgh
            wcal = math.floor(k * imgw)
            imgresize = cv2.resize(img, (wcal, imgsize))
            wgap = math.floor((imgsize - wcal) / 2)
            imgwhite[:, wgap:wcal + wgap] = imgresize

        else:
            k = imgsize / imgw
            hcal = math.floor(k * imgh)
            imgresize = cv2.resize(img, (imgsize, hcal))
            hgap = math.floor((imgsize - hcal) / 2)
            imgwhite[hgap:hcal + hgap, :] = imgresize
        return imgwhite
    #predicting the value from image
    def prediction(img):
        #calling resize_img function
        m = resize_img(img, len(img), len(img[0]))
        #normalization
        m = m / 255
        #(1,80,80,3)
        m = np.array([m])
        s = model.predict(m, verbose=0)
        return np.argmax(s)

    def camera():
        #open webcam
        cap = cv2.VideoCapture(0)
        #hand detector
        detector = HandDetector(maxHands=1)
        #additional pixels
        offset = 20
        #time control
        prev = time.time()
        #emplty string for calculation
        string = ''

        #Button column CSS
        st.markdown(
            '''<style>
                div[data-testid="column"]:nth-of-type(1){
                    color: red;
                    text-align : right;
                }
                div[data-testid="column"]:nth-of-type(2){
                    color: yellow;

                }    
                </style>
            ''', unsafe_allow_html=True
        )
        #STOP and REFRESH
        col1, col2 = st.columns(2)
        with col1:

            stop_button_pressed = st.button("Stop")
        with col2:
            refresh = st.button("Refresh")

        #WEBCAM and RESULT
        col1,col2 = st.columns(2)
        with col1:
            #initializing the webcam place
            frame_placeholder = st.empty()
        with col2:
            #if refresh
            if refresh:
                string = ''
            #position of calculation string
            number = st.empty()
            #position of result string
            result1 = st.empty()
            #when camera is open
            while cap.isOpened() and not stop_button_pressed:
                #read the image
                success, img = cap.read()
                #detect the 1st hand
                hands, img = detector.findHands(img)
                #if webcam not opened
                if not success:
                    st.write("Video Capture Ended")
                    break
                #when hand is found
                if hands:
                    hand = hands[0]
                    x, y, w, h = hand['bbox']

                    imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

                    cur = time.time()
                    if cur - prev > 3:
                        prev = cur
                        result = prediction(imgCrop)
                        if len(string) != 0 and result == 0:
                            if string[len(string) - 1] in ['+', '-', '*', '/']:
                                pass
                            else:
                                string += str(result)
                        elif string == '' and (
                                result == 0 or result == 10 or result == 11 or result == 12 or result == 13):
                            string = ''
                        elif result == 10:
                            string += '+'
                        elif result == 11:
                            string += '-'
                        elif result == 12:
                            string += '*'
                        elif result == 13:
                            string += '/'
                        elif string == '' and result == 15:
                            string = ''
                        elif string != '' and result == 15:  # DELETE
                            string = string[:len(string) - 1]
                        elif string == '' and result == 14:
                            st.title(string)
                            break
                        elif string != '' and result == 14:
                            #result
                            with result1.container():
                                val = str(eval(string))
                                st.markdown(
                                    f"<h3 style='text-align: center; color: yellow;'>Result: </h3>",
                                    unsafe_allow_html=True)
                                st.markdown(
                                    f"<h4 style='text-align: center; color: green; background-color: black;'>{val}</h4>",
                                    unsafe_allow_html=True)
                        else:
                            string += str(result)
                    #calucation string
                    with number.container():
                        st.markdown(
                            f"<h5 style='text-align: center; color: blue;'>You have entered: </h5>",
                            unsafe_allow_html=True)
                        st.markdown(f"<h6 style='text-align: center; color: white;'>{string}</h6>",
                                    unsafe_allow_html=True)

                frame_placeholder.image(img, channels="RGB")

                if cv2.waitKey(1) & 0xFF == ord("q") or stop_button_pressed:
                    break
            cap.release()
            cv2.destroyAllWindows()
    camera()
if __name__ == "__main__":
    main()