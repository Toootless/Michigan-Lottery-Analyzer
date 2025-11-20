from fixed_scraper import ProductionMichiganScraper
from src.analysis.mla3_predictor import predict

s = ProductionMichiganScraper()
h = s.get_recent_history("Powerball", 120)
print("History len:", len(h))
r = predict(h, "Powerball", n_sets=3)
print("Game:", r["game"])
for i, set_ in enumerate(r["sets"], 1):
    print(i, set_)
print("Explain:", r["explanations"][:2])
