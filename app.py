import streamlit as st
import gdown
import tensorflow as tf

def carrega_modelo():
    #https://drive.google.com/file/d/1GpWix8dp6FeFAs6g0etbnw_avu9Aflfp/view?usp=sharing
    url = 'https://drive.google.com/uc?id=1GpWix8dp6FeFAs6g0etbnw_avu9Aflfp'
    
    gdown.down(url, 'modelo_quantizado16bits.tflite')
    
    interpreter = tf.lite.Interpreter(model_path='modelo_quantizado16bits.tflite')
    
    interpreter.allocate_tensors()
    
    return interpreter
    

def main():
    st.set_page_config(
        page_title = "Classifica Folha de Videeiras"
    )
    
    st.write("#Classificação Folha de Videiras")
    
if __name__ == "__main__":
    main()