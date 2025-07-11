
## How to Run

1. **Collect Data:**  
   Use the provided script/notebook to fetch AAPL data from the Twelve Data API.

2. **Preprocess & Feature Engineering:**  
   Clean and prepare the data, generate features.

3. **Model Training:**  
   Train the models using scikit-learn (see `src/`).

4. **Deployment:**  
   Deploy the selected model to AWS SageMaker using the scripts in `deployment/`.

## Results

- Random Forest achieved lower RMSE and better R² than Linear Regression.
- Real-world test with AAPL data (July 7, 2025): Model predicted the next-day closing price within $0.33 accuracy.
- Full pipeline from data collection to cloud deployment.

## Live Demo / Code

The complete source code and project files are available here:  
[GitHub Repository](https://github.com/nasserpeiroti/Stock-Price-Prediction)

## Author

Nasser Peiroti  

## License

This project is for educational purposes.