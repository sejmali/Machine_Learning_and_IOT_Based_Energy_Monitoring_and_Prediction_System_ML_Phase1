
# Machine Learning and IoT-Based Energy Monitoring and Prediction System (Phase 1)

This repository contains the first phase of a Machine Learning project aimed at predicting energy consumption (in megawatts) for the next month based on historical hourly electricity usage data. The prediction is achieved using a Long Short-Term Memory (LSTM) neural network model, a type of Recurrent Neural Network (RNN) designed to handle time-series data efficiently.

## Project Overview

In this phase, we focus on training a model using historical energy consumption data and making predictions for the next 31 days. The dataset used is the **AEP Hourly** dataset, which records hourly power consumption in MW (megawatts).

The primary components of this project include:

- **Preprocessing the Data**: Handling missing values and scaling the data using MinMaxScaler to ensure all values lie between 0 and 1.
- **LSTM Model Development**: Building and training an LSTM model that can learn the temporal patterns in electricity consumption data.
- **Prediction Generation**: Predicting electricity consumption for the next 31 days (hourly intervals).
- **Visualization**: Displaying the predicted values using Seaborn and Matplotlib to visualize future energy consumption patterns.
- **Saving Results**: Storing the predictions as JSON files for easy retrieval and further analysis.

## Dataset

- **AEP Hourly Dataset**: This dataset consists of hourly electricity consumption data in megawatts (MW) for a specified region.
- The dataset is preprocessed by:
  - Converting the 'Datetime' column into a time-series index.
  - Handling missing values using forward filling.
  - Scaling the 'AEP_MW' column for efficient learning by the model.

## Key Steps in the Code

1. **Data Preprocessing**:
   - The dataset is loaded and transformed to make it ready for time-series prediction. This includes setting the time as the index and normalizing the energy consumption values.
   
2. **Time-Series Data Preparation**:
   - A sliding window of time steps is used to prepare the data. Each time step consists of 10 previous hourly readings used to predict the next one.

3. **LSTM Model**:
   - A Sequential LSTM model is built using TensorFlow/Keras. The model contains 50 LSTM units and a Dense layer with 1 unit for output prediction.
   - It is compiled using the Adam optimizer and mean squared error (MSE) as the loss function.
   
4. **Training**:
   - The model is trained using 80% of the dataset, with the remaining 20% used for validation. The training runs for 10 epochs with a batch size of 32.
   
5. **Prediction**:
   - Once trained, the model predicts electricity consumption for the next 31 days on an hourly basis.
   
6. **Visualization**:
   - The predicted values are visualized using Seaborn and Matplotlib. A plot of hourly energy consumption for the next month is displayed to show trends.
   
7. **Saving Predictions**:
   - The predicted results are saved as a JSON file, containing timestamps and predicted energy consumption values.

## Requirements

The following Python libraries are required to run the code:

- `numpy`
- `pandas`
- `seaborn`
- `tensorflow`
- `sklearn`
- `matplotlib`

You can install them using the given requirements.txt file:

```bash
pip install -r requirements.txt
```

## How to Run

1. Clone the repository:

```bash
git clone <repository-url>
```

2. Navigate to the project directory:

```bash
cd Machine_Learning_and_IOT_Based_Energy_Monitoring_and_Prediction_System_ML_Phase1
```

3. Run the code:

```bash
python <script_name>.py
```

4. After running the code, you will find the predicted power consumption stored in `future_predictions.json`. A plot showing the hourly prediction for the next month will also be displayed.

## Output

- **`future_predictions.json`**: Contains the predicted hourly energy consumption for the next 31 days.
- **Plot**: Displays the trends of predicted energy consumption over time.

## Conclusion

This project demonstrates the potential of machine learning in time-series forecasting. By analyzing historical electricity consumption data, the LSTM model helps predict future energy needs, which can be crucial for energy management and planning.


## Future Work

In the next phases, we aim to integrate this ML component with IoT modules for real-time monitoring and more accurate energy consumption predictions.
