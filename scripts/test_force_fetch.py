from fixed_scraper import ProductionMichiganScraper
import json

s = ProductionMichiganScraper()

results = {}
for game in ["Daily 4 Midday", "Daily 3 Midday", "Daily 4 Evening", "Daily 3 Evening"]:
    try:
        res = s.force_fetch_game(game)
    except Exception as e:
        res = {"success": False, "error": str(e)}
    results[game] = res

print(json.dumps(results, indent=2))
