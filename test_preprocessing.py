import unittest

import pandas as pd

from preprocessing import build_model_input


class PredictionEncodingTests(unittest.TestCase):
    def test_selected_categories_survive_single_row_encoding(self):
        features = ['Engine_SizeL', 'Make_BMW', 'Fuel_Type_Z']
        row = pd.DataFrame([{'Engine_SizeL': 2.0, 'Make': 'BMW', 'Fuel_Type': 'Z'}])
        result = build_model_input(row, ['Make', 'Fuel_Type'], features)
        self.assertEqual(result.iloc[0].tolist(), [2.0, 1, 1])

    def test_omitted_training_baseline_does_not_add_features(self):
        row = pd.DataFrame([{'Engine_SizeL': 1.6, 'Make': 'AUDI'}])
        result = build_model_input(row, ['Make'], ['Make_BMW', 'Engine_SizeL'])
        self.assertEqual(result.columns.tolist(), ['Make_BMW', 'Engine_SizeL'])
        self.assertEqual(result.iloc[0].tolist(), [0, 1.6])

    def test_changing_make_changes_prediction_features(self):
        rows = [pd.DataFrame([{'Make': make}]) for make in ['AUDI', 'BMW']]
        encoded = [build_model_input(row, ['Make'], ['Make_BMW']) for row in rows]
        self.assertEqual([frame.iloc[0, 0] for frame in encoded], [0, 1])


if __name__ == '__main__':
    unittest.main()
