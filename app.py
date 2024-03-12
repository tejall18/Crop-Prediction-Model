import streamlit as st 
import pandas as pd
import numpy as np
import os
import pickle
import warnings


# st.beta_set_page_config(page_title="Crop Recommender", page_icon="🌿", layout='centered', initial_sidebar_state="collapsed")

st.set_page_config(
	layout="centered",  # Can be "centered" or "wide". In the future also "dashboard", etc.
	initial_sidebar_state="collapsed",  # Can be "auto", "expanded", "collapsed"
	page_title="Crop Prediciton",  # String or None. Strings get appended with "• Streamlit". 
	page_icon="🌿",  # String, anything supported by st.image, or None.
)

# st.title('Yield Prediciton')
# st.('🌿')
# st.subheader('Training Data Stats')
# st.write(df.describe())

def load_model(modelfile):
	loaded_model = pickle.load(open(modelfile, 'rb'))
	return loaded_model

def main():
    # title
    html_temp = """
    <div>
    <h2 style="color:MEDIUMSEAGREEN;text-align:left;">Team: 20_Code Predators</h2>
    <h1 style="color:MEDIUMSEAGREEN;text-align:left;"> Crop Recommendation  🌱 </h1>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    col1,col2  = st.beta_columns([9.9,0.1])


    with col1:
        st.subheader(" Find out the most suitable crop to grow in your farm 👨‍🌾")
        N = st.number_input("Nitrogen", 1,10000)
        P = st.number_input("Phosporus", 1,10000)
        K = st.number_input("Potassium", 1,10000)
        temp = st.number_input("Temperature",0.0,100000.0)
        humidity = st.number_input("Humidity in %", 0.0,100000.0)
        ph = st.number_input("Ph", 0.0,100000.0)
        rainfall = st.number_input("Rainfall in mm",0.0,100000.0)

        feature_list = [N, P, K, temp, humidity, ph, rainfall]
        single_pred = np.array(feature_list).reshape(1,-1)
        
        if st.button('Predict'):

            loaded_model = load_model('model.pkl')
            prediction = loaded_model.predict(single_pred)
            col1.write('''
		    ## Results 🔍 
		    ''')
            col1.success(f"{prediction.item().title()} are recommended by the A.I for your farm.")
      #code for html ☘️ 🌾 🌳 👨‍🌾  🍃

    # st.warning("Note: This A.I application is for educational/demo purposes only and cannot be relied upon. Check the source code [here](https://github.com/gabbygab1233/Crop-Recommendation)")
    hide_menu_style = """
    <style>
    #MainMenu {visibility: hidden;}
    </style>
    """

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

if __name__ == '__main__':
	main()