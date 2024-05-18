import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler


model1 = load_model('lstm_model.h5')
model2 = load_model('stacked_model.h5')
model3 = load_model('bidirectional_model.h5')

# Define a function to prepare data for prediction
def prepare_data(df):
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)
    return scaled_data, scaler

# Define a function to predict energy consumption for the specified time using multiple models
def predict_energy_consumption(input_data, scaler, time):
    # Prepare the input data
    prepared_input = input_data.reshape((1, input_data.shape[0], input_data.shape[1]))
    
    # Predict energy consumption using each model
    predictions_model1 = model1.predict(prepared_input)
    predictions_model2 = model2.predict(prepared_input)
    predictions_model3 = model3.predict(prepared_input)
    
    # Inverse transform the predictions to get the original scale of energy consumption
    predictions_model1 = scaler.inverse_transform(predictions_model1)
    predictions_model2 = scaler.inverse_transform(predictions_model2)
    predictions_model3 = scaler.inverse_transform(predictions_model3)
    
    # Aggregate the predictions from all models
    predictions_combined = np.mean([predictions_model1, predictions_model2, predictions_model3], axis=0)
    
    # Calculate total energy consumption
    total_energy_consumption = np.sum(predictions_combined) * time
    
    # Calculate cost based on energy consumption (assuming 12 PHP per kWh)
    cost = total_energy_consumption * 12
    
    # Calculate time in hours
    time_hours = time   
    
    return predictions_combined, total_energy_consumption, cost, time_hours

# Streamlit app
def main():
    st.title('Smart Energy Management App')

    st.write('Enter Appliance Data:')
    
    # Input fields for each appliance
    appliance_data = []
    for i in range(3):
        st.write('## Appliance {}:'.format(i+1))
        energy_consumption = st.number_input('Energy Consumption (kWh) for Appliance {}'.format(i+1), min_value=0.0, step=0.01)
        appliance_data.append(energy_consumption)
    
    # Input field for time (in hours)
    time = st.number_input('Enter Time (in hours)', min_value=0.0, step=0.01)
    
    # Prepare input data for prediction
    input_data = np.array([appliance_data])
    scaled_input_data, scaler = prepare_data(input_data)

    # Button to trigger prediction
    if st.button('Predict Energy Consumption'):
        # Predict energy consumption
        predictions, total_energy_consumption, cost, time_hours = predict_energy_consumption(scaled_input_data, scaler, time)

        # Display the predictions
        st.write("## Energy Consumption Prediction:")
        for i, pred in enumerate(predictions[0]):
            st.write("Appliance {}: {:.2f} kWh".format(i+1, pred))
        
        st.write("Total Energy Consumption: {:.2f} kWh".format(total_energy_consumption))
        st.write("Time: {:.2f} hours".format(time_hours))
        st.write("Cost: PHP {:.2f}".format(cost))

if __name__ == "__main__":
    main()
