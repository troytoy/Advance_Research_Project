# GA-ML Trading Optimization Project

## Objective
To optimize trading strategy parameters using Genetic Algorithms (GA) and subsequently filter false signals using Machine Learning models (Random Forest or LSTM).

## Workflow
1. **Strategy Definition**: Define the trading strategy and its parameters.
2. **GA Optimization**: Use GA to find the best parameter set that maximizes a fitness function (e.g., Sharpe Ratio, Total Return).
3. **Data Labeling**: Generate signals using the optimized strategy and label them (True/False positive) based on future outcomes.
4. **ML Filtering**: Train a classifier (RF or LSTM) to predict whether a signal is profitable.

## Tech Stack
- **GA**: DEAP or PyGAD
- **ML**: Scikit-Learn (Random Forest), TensorFlow/PyTorch (LSTM)
- **Backtesting**: Backtrader or VectorBT (to be decided)
