"""Quick sanity test for strategy uniqueness after diversification patch."""
from MichiganLotteryAnalyzer import analyze_frequency, generate_number_suggestions, LOTTERY_GAMES

# Synthetic historical results for a large-range game (Powerball style)
# Minimal sample to force sparse/overlapping hot/cold sets
sample_results = [
    {'numbers': [1, 5, 12, 23, 34, 7]},
    {'numbers': [2, 5, 14, 23, 36, 11]},
    {'numbers': [1, 8, 12, 25, 34, 22]},
    {'numbers': [3, 5, 16, 23, 34, 9]},
    {'numbers': [4, 6, 12, 27, 33, 18]},
    {'numbers': [1, 5, 12, 23, 34, 15]},
]

analysis = analyze_frequency(sample_results, 'Powerball')
strategies = generate_number_suggestions(analysis, 'Powerball')

print("Generated strategies (Powerball):")
for name, data in strategies.items():
    print(f"{name:18s} -> {data['numbers']}")

# Simple uniqueness assertion
sets = [tuple(v['numbers']) for v in strategies.values()]
print("\nUniqueness check:")
if len(sets) == len(set(sets)):
    print("PASS: All strategy number sets are unique.")
else:
    print("FAIL: Duplicate strategy sets detected.")
