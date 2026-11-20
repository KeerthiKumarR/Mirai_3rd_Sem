#!/usr/bin/env python3
"""
Unit tests and evaluation suite for Project 8 Assessment Questions.
"""

import unittest
from main import answer

class TestCulinaryRAG(unittest.TestCase):

    def test_q1_signature_truffle_ganache(self):
        question = "What exact quantities, in grams, of dark couverture chocolate and heavy cream are required for a standard 20-portion batch of Signature Truffle Ganache?"
        result = answer(question)
        
        self.assertIsInstance(result, dict)
        self.assertIn("answer", result)
        self.assertIn("sources", result)
        
        ans = result["answer"].lower()
        self.assertIn("850", ans)
        self.assertIn("620", ans)
        self.assertTrue("gram" in ans or "850 g" in ans or "850 g" in ans or "850g" in ans)
        
        # Verify source reference
        self.assertTrue(any("10" in str(s) for s in result["sources"]))

    def test_q2_autumn_tasting_menu_allergens(self):
        question = "According to the Allergen Matrix, which dishes on the Autumn Tasting Menu contain tree nuts or sesame?"
        result = answer(question)
        
        self.assertIsInstance(result, dict)
        self.assertIn("answer", result)
        self.assertIn("sources", result)
        
        ans = result["answer"]
        # Tree nuts dishes: Roasted Fig & Hazelnut Duck Terrine, Pistachio Crusted Lamb Rack
        self.assertIn("Roasted Fig", ans)
        self.assertIn("Pistachio Crusted Lamb Rack", ans)
        # Sesame dish: Charred Eggplant & Tahini Veloute
        self.assertIn("Charred Eggplant", ans)
        
        # Verify source reference
        self.assertTrue(any("25" in str(s) for s in result["sources"]))

    def test_q3_eiffel_tower_winter_hours(self):
        question = "At what time does the Eiffel Tower in Paris close to visitors during the winter season?"
        result = answer(question)
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result["answer"], "[not found]")
        self.assertEqual(result["sources"], [])

    def test_q4_sous_vide_duck_breast_food_safety(self):
        question = "What is the mandatory minimum internal cooking temperature, in degrees Celsius, and the required holding time for sous-vide duck breast, as specified in the Food Safety section?"
        result = answer(question)
        
        self.assertIsInstance(result, dict)
        self.assertIn("answer", result)
        self.assertIn("sources", result)
        
        ans = result["answer"]
        # Temperature: 57.5 degrees Celsius
        self.assertIn("57.5", ans)
        # Holding time: 110 minutes
        self.assertIn("110", ans)
        self.assertTrue("minute" in ans.lower() or "min" in ans.lower())
        
        # Verify source reference
        self.assertTrue(any("40" in str(s) for s in result["sources"]))

    def test_q5_noma_michelin_rating(self):
        question = "What is the current Michelin-star rating of Noma in Copenhagen?"
        result = answer(question)
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result["answer"], "[not found]")
        self.assertEqual(result["sources"], [])

if __name__ == "__main__":
    unittest.main(verbosity=2)
