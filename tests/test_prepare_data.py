import unittest

from scripts.prepare_data import make_housing_dataset


class MakeHousingDatasetTests(unittest.TestCase):
    def test_same_seed_produces_same_rows(self):
        first = make_housing_dataset(sample_count=25, seed=7)
        second = make_housing_dataset(sample_count=25, seed=7)
        self.assertTrue(first.equals(second))

    def test_raw_schema_contains_target_and_expected_features(self):
        frame = make_housing_dataset(sample_count=25, seed=7)
        self.assertEqual(
            list(frame.columns),
            [
                "area_sqft",
                "bedrooms",
                "bathrooms",
                "house_age",
                "distance_km",
                "price_usd",
                "garage_spots",
            ],
        )
        self.assertEqual(len(frame), 25)


if __name__ == "__main__":
    unittest.main()