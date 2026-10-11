
import pandas as pd
from sklearn.linear_model import LinearRegression


def load_data(filename="Salary Data.xlsx"):
    data = pd.read_excel(filename)
    return data


def train_model(data):
    X = data[["Years of Experience"]]
    y = data["Salary"]

    model = LinearRegression()
    model.fit(X, y)
    return model


def predict_salary(model, experience):
    if experience < 0:
        raise ValueError("Experience cannot be negative")

    prediction = model.predict(
        pd.DataFrame(
            {"Years of Experience": [experience]}
        )
    )
    return float(prediction[0])


if __name__ == "__main__":
    data = load_data()
    model = train_model(data)

    salary = predict_salary(model, 5)
    print("Predicted salary:", round(salary, 2))
