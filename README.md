# **Physics-Encoded Deep Learning for Uncovering Hidden Operational Patterns of Large Reservoirs**

# 1. Overview

This repository contains the PyTorch implementation of the DL-Res model, as proposed in our recent research work.

For information regarding the usage and distribution of this code, please refer to the `LICENSE` file included in this repository.

Should you encounter any issues, have suggestions for improvement, or identify any bugs, we encourage you to contact us at *12232254@mail.sustech.edu.cn*.



# 2. What's in this repository?

- `code` folder: all scripts to run DL-Res model
  - `main.py`: This Python script serves as the entry point for training and evaluating the DL-Res model. It sets up command-line arguments for configuring the model and training process, initializes the model, loads and preprocesses the data, and runs multiple experiments with early stopping and learning rate scheduling. The script supports CUDA acceleration and provides reproducibility through setting a fixed random seed.
  - `preparedataset.py`: features. The `traindata_watershed` and `traindata_reservoir` functions prepare data specifically for the Watershed and Reservoir LSTM components, respectively, by splitting the data into training, validation, and testing sets, applying normalization, and wrapping the data for LSTM input. Additionally, the `plot_data_preprocess_phy` function prepares data for plotting in a single batch for visualization purposes.
  - `data_utils.py`: This file provides utility functions for generating training, validation, and test datasets for the DL-Res model. It includes two main functions: `generate_train_val_test_phy` for preparing data for the Reservoir LSTM component, which wraps data into sequences with a specified length and handles time-series splitting, and `get_wrapped_data` for the Watershed LSTM component, which uses a sliding window approach to create input and target sequences. These functions ensure the data is appropriately formatted for training and evaluation.
  - `utils.py`: This file provides utility functions for the DL-Res model, including the calculation of Nash-Sutcliffe Efficiency (NSE) loss for both batched and non-batched data (`NSE_loss_BTF`, `NSE_loss_BF`), saving and reloading the model along with its arguments (`save_model`), setting the random seed for reproducibility (`set_seed`), and calculating performance metrics such as NSE, Pearson correlation, Kling-Gupta efficiency (KGE), and root mean squared error (RMSE) (`cal_nse`, `cal_metrics`). These utilities support model evaluation and ensure consistent results across runs.
  - `models.py`: This file contains the definitions of the DL-Res model and its base class. The `Base_Model` class provides a template for DL-Res models and sets up common parameters and optimizer configurations. The `DL_Res_LSTM` class extends `Base_Model` and consists of two main components: a Watershed LSTM for predicting inflow and a Reservoir LSTM for predicting releases based on the predicted inflow and other inputs. It includes methods for initializing the model, configuring optimizers, and defining the forward pass.
  - `layers.py`: This file contains the custom neural network layers used in the DL-Res model. It includes the `Watershed_LSTM_Layer`, a standard LSTM layer designed for predicting inflow into the watershed, incorporating dropout for regularization and a readout layer with a GELU activation function. The `Reservoir_LSTM_Layer_Jit` is an LSTM layer implemented using PyTorch JIT for the reservoir component, featuring a reservoir bucket model for simulating water balance, along with specific reservoir properties such as storage and release constraints. It also utilizes a custom Heaviside function for smooth step transitions. Lastly, the `PolynomialDecayLR` is a custom learning rate scheduler that implements polynomial decay with an optional warm-up phase, used to gradually decrease the learning rate over the course of training.
  - `train_utils.py`: This file contains utility functions for training, validating, and testing the DL-Res model. It includes methods for running the model on the test set (`test_model`), evaluating the model on the validation set (`val_model`), and training the model for one epoch (`train_epoch`). The `run_exp` function orchestrates the entire training process, including early stopping, saving the best model, and logging the results using TensorBoard. These functions enable efficient training, validation, and testing of the model, facilitating the analysis of its performance on hydrological data.
  - `plot_utils.py`: This file provides functions for creating evaluation plots to visualize the performance of the DL-Res model. It includes methods for generating single and full evaluation plots comparing observed and predicted values for inflow and release, as well as computing performance metrics such as NSE, Pearson correlation, RMSE, and KGE. The `plot_result` function orchestrates the entire evaluation process, from preprocessing data to plotting results for training, validation, and testing periods, ensuring comprehensive visual analysis of the model's performance.
- `requirements.txt`: This file contains the dependencies of our running environment.



# 3. How to set up the Python environment?

It is strongly recommended to use a virtual environment for this project. All dependencies can be installed from the `requirements.txt` file by executing the following command:

```shell
pip install -r requriements.txt
```



# 4. How to run the model?

## 4.1 Download and prepare data sets

To prepare the data sets for this study, follow these steps:

1. **Download Meteorological Data**: Obtain the daily precipitation and mean air temperature data from the 18 meteorological stations used in this study. These data can be accessed through the China Meteorological Data Sharing Service System (https://data.cma.cn/).
2. **Compute Basin-Averaged Time Series**: Compute the basin-averaged time series for these meteorological factors using the Thiessen polygon method.
3. **Acquire Reservoir Operation Data**: Retrieve the daily inflow, release, and storage data for the Ankang and Danjiangkou Reservoirs from the Bureau of Hydrology of the Yangtze Water Resources Commission of China ([http://www.cjh.com.cn](http://www.cjh.com.cn/)).
4. **Combine Data**: Place the time series of meteorological factors (precipitation, temperature) and the reservoir operation records in a single CSV file.

## 4.2 Run the model

1. **Modify File Paths**: Adjust the data file paths in the code to match the location of your prepared data files.
2. **Execute the Script**: Run the model by executing the script with the command `python main.py`.

## 5. Other information

- **References**: References to related papers will be added later.

