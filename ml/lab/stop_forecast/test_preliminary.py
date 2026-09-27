"""Input matching and quarter cutoff checks; fixtures are not research data."""
from datetime import date
from pathlib import Path
import tempfile
import unittest
from build_preliminary import named_metro, metro_history


class PreliminaryTest(unittest.TestCase):
    def test_explicit_names_and_time_cutoff(self):
        self.assertEqual(named_metro(' Метро «Семёновская» '), 'семеновская')
        self.assertEqual(named_metro('Метро "Проспект Мира"'), 'проспект мира')
        self.assertIsNone(named_metro('Метрополитен — служебная остановка'))
        self.assertIsNone(named_metro('Семёновская площадь'))
        text = ('NameOfStation;Line;Year;Quarter;IncomingPassengers;OutgoingPassengers;global_id\n'
                'служебная;строка;год;квартал;входы;выходы;ID\n'
                'Тест;A;2025;III квартал;10;0;1\n'
                'Тест;A;2025;IV квартал;99;88;2\n')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'fixture.csv'; path.write_text(text)
            frame = metro_history(path, date(2025, 10, 31))
            self.assertEqual(len(frame), 1)
            self.assertEqual(frame.iloc[0].IncomingPassengers, 10)
            self.assertTrue(frame.iloc[0].zero_counter_requires_review)
            self.assertFalse(frame.iloc[0].strict_asof_availability_verified)
            self.assertTrue(metro_history(path, date(2025, 8, 31)).empty)


if __name__ == '__main__':
    unittest.main()
