This project consists of 3 main functions(each running in threads):

1. Synthetic data generation - This function simulates the energy prices every hour, starting from 1 Jan 2024 adding components like drift and seasonality to the original data.
2. Data fetch, model training and inferencing - This function reads the synthetic data, trains the model and loads that trained model for inferencing on future data points.
3. Data visualisation - This function visualises the data and highlights anomalous instances.

### Model Chosen:

Isolation Forest was chosen for this particular task due to the following reasons:

1. Unsupervised learning algorithm specifically designed for anomaly detection.
2. Rapid training and inferencing.
3. Highly robust and capable of capturing fine patterns in the data.
