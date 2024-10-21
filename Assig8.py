import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import json
import os
import unittest



# Step 1: Data Manipulation (Loading and Cleaning)
try:
    # Load the dataset from the specified CSV file
    data = pd.read_csv("C:/Users/user/Desktop/Day 1/week8_python_project/housing.csv")
    
    # Preprocess data by filling missing values using forward fill method
    data.ffill(inplace=True)  # Avoids deprecation warning
except FileNotFoundError:
    # Handle the case where the file is not found
    print("File not found!")
    data = None  # Set data to None if the file is not found



# Step 2: Data Visualization
if data is not None:
    sns.pairplot(data)
    plt.show()  # Display the plots
else:
    print("Data not available for visualization.")



# Step 3: Machine Learning Model
if data is not None:
    # Define features (X) and target variable (y)
    X = data[['Area', 'Bedrooms', 'Location_Score']]
    y = data['Price']

    # Split the dataset into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Initialize the Linear Regression model
    model = LinearRegression()
    
    # Train the model using the training data
    model.fit(X_train, y_train)
    
    # Make predictions using the test data
    predictions = model.predict(X_test)



    # Step 4: Evaluation and File I/O
    '''Calculate Mean Squared Error (MSE) to evaluate model performance'''
    mse = mean_squared_error(y_test, predictions)
    results = {'Mean Squared Error': mse}  # Store results in a dictionary

    # Ensure the 'output' directory exists before saving files
    output_dir = 'C:/Users/user/Desktop/Day 1/week8_python_project/output'
    os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

    # Save the evaluation results to a JSON file in the output folder
    with open(f'{output_dir}/results.json', 'w') as f:
        json.dump(results, f)

    # Save predictions to CSV
    try:
        with open(f'{output_dir}/predictions.csv', 'w') as f:
            pd.DataFrame(predictions, columns=['Predicted Price']).to_csv(f)
    except IOError:
        print("An error occurred while writing the file.")
else:
    print("Data not available for model training.")




# Step 5: Testing (Using unittest)
class TestHousingModel(unittest.TestCase):
    def test_data_loading(self):
        # Test that the data loading was successful (data should not be empty)
        self.assertFalse(data.empty, "Data loading failed!")
    
    def test_model_output(self):
        # Test that predictions were made (predictions list should not be empty)
        self.assertTrue(len(predictions) > 0, "No predictions made!")

# Entry point for running unit tests
if __name__ == "__main__":
    unittest.main()
