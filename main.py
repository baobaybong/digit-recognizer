import pandas as pd
import streamlit as st
from streamlit_drawable_canvas import st_canvas

from utils import predict, format, CNN

from PIL import Image
import numpy as np
import os

def main():
    st.title("Digit recognizer🎨🖌️")

    col1, col2 = st.columns(2)

    with col1:
        canvas_result = st_canvas(
            background_color="#FFFFFF",
            height=300,
            width=300,
        )
    
    with col2:
        num_models = 2 if st.checkbox("Compare two models") else 1

        cols = st.columns(num_models)

        for col in cols:
            with col:
                models = os.listdir("models")
                model_selected = st.selectbox(
                    "Choose model",
                    models,
                    format_func=lambda x: x.split(".")[0],  
                    key=col
                )
                
                if canvas_result.image_data is not None:
                    img = np.array(format(canvas_result.image_data))[0] * 255
                    st.image(Image.fromarray(img).convert('L'))

                    st.write("Predicted", predict(canvas_result.image_data, model_selected))
                    
                    # st.write(predict(canvas_result.image_data)[0])
                    # if st.button("Save Image"):
                    #     image = Image.fromarray(img).convert('L')
                    #     image.save("saved_image.png")
                    #     st.success("Image saved successfully!")
                else:
                    st.write("lol")

if __name__ == "__main__":
    main()