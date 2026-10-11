
import pandas as pd
from sklearn.linear_model import LinearRegression


def load_data(filename="Salary Data.xlsx"):
    data = pd.read_excel(filename)

    required_columns = ["Years of Experience", "Salary"]

    for column in required_columns:
        if column not in data.columns:
            raise ValueError(f"Missing column: {column}")

        data[column] = pd.to_numeric(
            data[column], errors="coerce"
        )

    # Remove rows with missing or invalid values
    data = data.dropna(subset=required_columns)

    if data.empty:
        raise ValueError("No valid salary data available")

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
        pd.DataFrame({
            "Years of Experience": [experience]
        })
    )

    return float(prediction[0])


if __name__ == "__main__":
    data = load_data()
    model = train_model(data)

    salary = predict_salary(model, 5)

    print("Valid dataset rows:", len(data))
    print("Predicted salary:", round(salary, 2))
