import json, random, os
src = "MP_json_is_metal/id_prop.json"
rows = json.load(open(src))
random.Random(0).shuffle(rows)
sub = rows[:3000]
lbl = [int(r["is_metal"]) for r in sub]
os.makedirs("MP_json_is_metal_smoke", exist_ok=True)
json.dump(sub, open("MP_json_is_metal_smoke/id_prop.json", "w"))
print(f"wrote {len(sub)} records; metal={sum(lbl)} nonmetal={len(lbl)-sum(lbl)}")
