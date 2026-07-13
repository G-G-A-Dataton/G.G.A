import unittest

import pandas as pd

from src.attributes import add_attribute_features, parse_attributes, parse_color
from src.features import (
    FEATURE_COLS,
    build_features,
    compute_age_group_match,
    compute_gender_match,
    compute_query_brand_match,
    compute_cat_depth,
    tokenize,
)
from src.item_text import parse_attributes_flat


class AttributeParsingTests(unittest.TestCase):
    def test_parses_real_flat_catalog_format(self):
        raw = (
            "materyal: tekstil, renk: gri, "
            "materyal bileşeni: astar: 100% polyester, ortam: casual/günlük"
        )
        parsed = parse_attributes(raw)
        self.assertEqual(parsed["materyal"], "tekstil")
        self.assertEqual(parsed["renk"], "gri")
        self.assertEqual(parsed["materyal bileşeni"], "astar: 100% polyester")
        self.assertEqual(parse_color(raw), "gri")
        self.assertIn("materyal tekstil", parse_attributes_flat(raw))

    def test_retains_dict_and_json_compatibility(self):
        self.assertEqual(parse_attributes("{'Renk': 'Siyah'}"), {"renk": "siyah"})
        self.assertEqual(parse_attributes('{"Size": "42"}'), {"size": "42"})

    def test_attribute_builder_matches_query_values(self):
        frame = pd.DataFrame(
            [
                {
                    "item_id": "i1",
                    "query": "gri tekstil çanta",
                    "attributes": "materyal: tekstil, renk: gri",
                },
                {
                    "item_id": "i1",
                    "query": "beyaz deri çanta",
                    "attributes": "materyal: tekstil, renk: gri",
                },
            ]
        )
        result = add_attribute_features(frame)
        self.assertEqual(result["query_color_match"].tolist(), [1, -1])
        self.assertEqual(result["query_material_match"].tolist(), [1, -1])


class ExactTextMatchingTests(unittest.TestCase):
    def test_brand_requires_complete_words(self):
        self.assertEqual(compute_query_brand_match("samsung telefon", "sam"), 0)
        self.assertEqual(compute_query_brand_match("sam telefon", "sam"), 1)
        self.assertEqual(compute_query_brand_match("m&s kadın giyim", "M&S"), 1)

    def test_demographic_keywords_require_complete_words(self):
        self.assertEqual(compute_gender_match("bayram şekeri", "erkek"), 0)
        self.assertEqual(compute_gender_match("bayan çanta", "erkek"), -1)
        self.assertEqual(compute_age_group_match("kidney ürünü", "çocuk"), 0)
        self.assertEqual(compute_age_group_match("kid oyuncak", "çocuk"), 1)
        self.assertEqual(
            compute_age_group_match("bebek çocuk body", "bebek & çocuk"), 1
        )

    def test_feature_builder_preserves_contract(self):
        frame = pd.DataFrame(
            [
                {
                    "item_id": "i1",
                    "query": "gri tekstil çanta",
                    "title": "Gri Çanta",
                    "category": "aksesuar/çanta",
                    "brand": "Acme",
                    "gender": "unisex",
                    "age_group": "yetişkin",
                    "attributes": "materyal: tekstil, renk: gri",
                }
            ]
        )
        result = build_features(frame)
        self.assertEqual(result[FEATURE_COLS].columns.tolist(), FEATURE_COLS)
        self.assertEqual(result.loc[0, "query_color_match"], 1)
        self.assertEqual(result.loc[0, "query_material_match"], 1)

    def test_lexical_features_normalize_punctuation_and_model_codes(self):
        frame = pd.DataFrame(
            [
                {
                    "item_id": "i1",
                    "query": "Samsung SM-A536B telefon",
                    "title": "Samsung Galaxy SM A536B Telefon",
                    "category": "Elektronik/Cep Telefonu",
                    "brand": "Samsung",
                    "gender": "unisex",
                    "age_group": "yetişkin",
                    "attributes": "",
                },
                {
                    "item_id": "i2",
                    "query": "Samsung A536B telefon",
                    "title": "Samsung Galaxy A525F Telefon",
                    "category": "Elektronik/Cep Telefonu",
                    "brand": "Samsung",
                    "gender": "unisex",
                    "age_group": "yetişkin",
                    "attributes": "",
                },
            ]
        )
        result = build_features(frame, verbose=False)
        self.assertIn("a536b", tokenize("SM-A536B"))
        self.assertEqual(result["query_model_token_match"].tolist(), [1, 0])
        self.assertEqual(result["query_model_token_conflict"].tolist(), [0, 1])
        self.assertGreater(result.loc[0, "query_title_coverage"], 0.5)

    def test_empty_category_has_zero_depth(self):
        self.assertEqual(compute_cat_depth(""), 0)
        self.assertEqual(compute_cat_depth(None), 0)


if __name__ == "__main__":
    unittest.main()
