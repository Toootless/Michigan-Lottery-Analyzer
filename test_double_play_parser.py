#!/usr/bin/env python3
"""Regression tests for structured Powerball double play parsing.
Run directly or via a test runner.
"""
from typing import Dict, Any
from bs4 import BeautifulSoup
from fixed_scraper import ProductionMichiganScraper

POWERBALL_FIXTURE = """
<table>
<tr class="c-results-table__item">
  <div class="c-draw-card__ball-box">
    <ul>
      <li class="c-ball">29</li>
      <li class="c-ball">39</li>
      <li class="c-ball">43</li>
      <li class="c-ball">51</li>
      <li class="c-ball">65</li>
      <li class="c-result__bonus"><span class="c-ball">23</span></li>
      <li class="c-result__multiplier">Power Play: 2</li>
    </ul>
  </div>
  <div class="c-draw-card__ball-box">
    <span class="c-draw-card__ball-title">Double Play</span>
    <ul>
      <li class="c-ball">7</li>
      <li class="c-ball">31</li>
      <li class="c-ball">32</li>
      <li class="c-ball">39</li>
      <li class="c-ball">65</li>
      <li class="c-result__bonus"><span class="c-ball">11</span></li>
      <li class="c-result__multiplier">Double Play Multiplier: 3</li>
    </ul>
  </div>
  <div class="c-draw-card__draw-date-dow">Wednesday,</div>
  <div class="c-draw-card__draw-date-sub">Nov 12, 2025</div>
</tr>
</table>
"""

# Fixture without double play multiplier
POWERBALL_FIXTURE_NO_DPM = POWERBALL_FIXTURE.replace('Double Play Multiplier: 3', '')


def parse_fixture(html: str) -> Dict[str, Any]:
    scraper = ProductionMichiganScraper()
    soup = BeautifulSoup(html, 'html.parser')
    structured = scraper._parse_lotteryusa_structured(soup, 'Powerball')  # type: ignore[attr-defined]
    if not structured:
        raise AssertionError('Structured parser returned None')
    return structured


def test_powerball_double_play_full():
    data = parse_fixture(POWERBALL_FIXTURE)
    assert data['main_numbers'] == [29, 39, 43, 51, 65], 'Main numbers mismatch'
    assert data['bonus_number'] == 23, 'Bonus number mismatch'
    assert data['double_play_numbers'] == [7, 31, 32, 39, 65], 'Double Play numbers mismatch'
    assert data['double_play_bonus'] == 11, 'Double Play bonus mismatch'
    assert 'multiplier' in data and data['multiplier'] == 'Power Play: 2', 'Power Play missing or incorrect'
    assert 'double_play_multiplier' in data and data['double_play_multiplier'] == 'Double Play Multiplier: 3', 'Double Play Multiplier missing'


def test_powerball_double_play_without_multiplier():
    data = parse_fixture(POWERBALL_FIXTURE_NO_DPM)
    assert data['main_numbers'] == [29, 39, 43, 51, 65]
    assert data['bonus_number'] == 23
    assert data['double_play_numbers'] == [7, 31, 32, 39, 65]
    assert data['double_play_bonus'] == 11
    assert 'multiplier' in data and data['multiplier'] == 'Power Play: 2'
    assert 'double_play_multiplier' not in data, 'Double Play multiplier should be absent'


def run_all():
    failures = []
    for fn in [test_powerball_double_play_full, test_powerball_double_play_without_multiplier]:
        try:
            fn()
            print(f"✅ {fn.__name__} passed")
        except AssertionError as e:
            failures.append(f"❌ {fn.__name__}: {e}")
    if failures:
        print('\nTest Failures:')
        for f in failures:
            print(f)
        raise SystemExit(1)
    print('\nAll double play parser tests passed.')

if __name__ == '__main__':
    run_all()
