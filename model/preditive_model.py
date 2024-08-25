from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

from utils import data_cleaning

def model():
    features, target = data_cleaning.cleaning()
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    # visualize
    visualization(y_test, predictions)
    print(f'MSE: {mse}, R²: {r2}')


def visualization(y_test, predictions):
    plt.scatter(y_test, predictions)
    plt.xlabel('Actual Emissions')
    plt.ylabel('Predicted Emissions')
    plt.show()


if __name__ == "__main__":
    model()