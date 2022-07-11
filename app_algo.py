import streamlit as st
from multiapp import MultiApp

def app():
    container = st.empty()
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<h4>Baseline Algorithm</h4>", unsafe_allow_html=True)
        st.write('Deep Stride Convolutional Neural Network')
        st.write('Deep Stride Convolutional Neural Networks (DSCNN) architecture is known for its innovative results in image recognition, objects detection, image classification, and other areas in the field of computer vision.')
        st.write('This algorithm utilizes SoftMax activation function that performs as a classifier. The algorithm includes one pre-processing layer allotted for applying the short-term Fourier transform (STFT) to the speech signal to obtain the conventional spectrograms.')
        st.write('It has five (5) convolutional layers, a flatten layer and two (2) fully connected layers.')
    with col2:
        st.markdown("<h4>Modified Algorithm</h4>", unsafe_allow_html=True)
        st.write('Deep Stride Convolutional Neural Network with Extreme Learning Machine Algorithm')
        st.write('Modified algorithm uses replaces the end layer of the baseline neural network architecture. Instead of using the SoftMax classifier, another neural network is applied. This neural network accept the output of the flatten layer of the baseline algorithm.')
        st.write('Moreover, Extreme Learning Machine (ELM) is a single-layer neural network with a feedforward approach. It is a fast-learning algorithm whose implementation is commonly used for classification, clustering, and regression.')
        st.write('It has five (5) convolutional layers, a flatten layer, two (2) fully connected layers, and an ELM classifier algorithm.')