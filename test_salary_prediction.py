
import unittest
import pandas as pd

from salary_prediction import (
    load_data,
    train_model,
    predict_salary
)


class TestSalaryPrediction(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data = load_data()
        cls.model = train_model(cls.data)

    def test_dataset_loaded(self):
        self.assertFalse(self.data.empty)

    def test_salary_column(self):
        self.assertIn("Salary", self.data.columns)

    def test_experience_column(self):
        self.assertIn(
            "Years of Experience",
            self.data.columns
        )

    def test_salary_prediction(self):
        result = predict_salary(self.model, 5)
        self.assertIsInstance(result, float)

    def test_negative_experience(self):
        with self.assertRaises(ValueError):
            predict_salary(self.model, -2)


if __name__ == "__main__":
    unittest.main()
