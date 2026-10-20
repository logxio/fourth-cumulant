"""Check the released Phase 2 submission-page readings against the number table."""

import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PAGES = ROOT / "data/submission_pages_1019"
ENTRIES = (
    ("us", "336537", "Susupro"),
    ("chainb", "335301", "Susupro"),
    ("hyojun", "335413", "hyojun_kwon"),
    ("artyom", "333534", "Artyom_Mazur"),
    ("nurlan", "336225", "Nurlanggs"),
    ("miscusi", "336317", "il.miscusi"),
    ("kai", "334821", "kai_leal_gonzalez"),
    ("aldo", "330093", "504aldo"),
    ("ddteam", "335813", "DD_team"),
)


def readings(numbers):
    values = {}
    for tag, sid, author in ENTRIES:
        page = (PAGES / f"{sid}.md").read_text()
        found = re.search(r"Phase 2 · (.+?)whestbench\.submission\." + sid, page)
        assert found and found.group(1).replace("\\_", "_") == author, sid
        adj = re.search(r"Adjusted score\s+Public split\s+([0-9.]+×10−\d+)", page)
        raw = re.search(r"Final-layer MSE\s+([0-9.]+×10−\d+)", page)
        cb = re.search(r"Budget used\s+([0-9.]+)%", page)
        assert adj and raw and cb, sid
        values[tag] = {"adj": float(adj.group(1).replace("×10−", "e-")),
                       "raw": float(raw.group(1).replace("×10−", "e-")),
                       "cb": float(cb.group(1)) / 100}
        for field, actual in values[tag].items():
            expected = numbers[f"nb_{tag}_{field}"]
            assert abs(actual / expected - 1) < 5e-6, (tag, field)
    public = json.loads((ROOT / "report/data/res_336537.json").read_text())["aggregates"]["public"]
    assert values["us"]["adj"] == public["adjusted_final_layer_score"]
    assert values["us"]["raw"] == public["final_layer_mse"]
    for tag, _, _ in ENTRIES[1:]:
        for field in ("adj", "raw", "cb"):
            actual = values["us"][field] / values[tag][field]
            assert abs(actual / numbers[f"nb_us_over_{tag}_{field}"] - 1) < 6e-6
    for tag, _, _ in ENTRIES[2:]:
        actual = values["chainb"]["adj"] / values[tag]["adj"]
        assert abs(actual / numbers[f"nb_chainb_over_{tag}_adj"] - 1) < 6e-6
    base = (PAGES / "334316.md").read_text()
    assert "hyojun_kwon" in base
    base_adj = re.search(r"Adjusted score\s+Public split\s+([0-9.]+×10−\d+)", base)
    base_raw = re.search(r"Final-layer MSE\s+([0-9.]+×10−\d+)", base)
    base_cb = re.search(r"Budget used\s+([0-9.]+)%", base)
    assert base_adj and base_raw and base_cb
    values["hyojun_base"] = {"adj": float(base_adj.group(1).replace("×10−", "e-")),
                             "raw": float(base_raw.group(1).replace("×10−", "e-")),
                             "cb": float(base_cb.group(1)) / 100}
    for field, actual in values["hyojun_base"].items():
        assert abs(actual / numbers[f"nb_hyojun_base_{field}"] - 1) < 5e-6
    for field in ("adj", "raw"):
        assert abs(values["chainb"][field] / values["hyojun_base"][field]
                   / numbers[f"nb_chainb_over_hyojun_base_{field}"] - 1) < 6e-6
    assert abs(values["us"]["adj"] / values["chainb"]["adj"]
               / numbers["nb_us_corrector_factor"] - 1) < 6e-6
    assert abs(values["hyojun"]["adj"] / values["hyojun_base"]["adj"]
               / numbers["nb_hyojun_corrector_factor"] - 1) < 6e-6
    others = sorted(ENTRIES[2:], key=lambda row: values[row[0]]["adj"])
    assert numbers["nb_writeups_n"] == 8
    assert numbers["nb_us_rank_of_writeups"] == 1 + sum(values[t]["adj"] < values["us"]["adj"] for t, _, _ in others)
    assert numbers["nb_chainb_rank_of_writeups"] == 1 + sum(values[t]["adj"] < values["chainb"]["adj"] for t, _, _ in others)
    assert numbers["nb_below_chainb_n"] == sum(values[t]["adj"] < values["chainb"]["adj"] for t, _, _ in ENTRIES if t != "chainb")
    assert numbers["nb_hyojun_cited_adj"] == 3.044e-9
    assert numbers["nb_e30a_adj"] == 3.116e-9
    in_class = [values[t]["adj"] for t in ("hyojun_base", "artyom", "nurlan", "kai", "aldo", "ddteam")]
    in_class += [1.9951e-8 * 0.267, values["hyojun_base"]["adj"] / (1 - 0.0122)]
    assert numbers["nb_chainb_rank_no_corrector_no_table"] == 1 + sum(a < values["chainb"]["adj"] for a in in_class)
    assert abs(numbers["nb_us_below_next_adj"] - (1 - values["us"]["adj"] / values[others[0][0]]["adj"])) < 1e-6
    assert abs(numbers["nb_us_below_miscusi_adj"] - (1 - values["us"]["adj"] / values["miscusi"]["adj"])) < 1e-6
    assert numbers["nb_lowest_raw_is_miscusi"] == int(min(values, key=lambda t: values[t]["raw"]) == "miscusi")
    with (ROOT / "data/leaderboard_1019_1330.csv").open(newline="") as handle:
        board = {row["submission"]: row for row in csv.DictReader(handle)}
    listed = set()
    for tag, sid, _ in ENTRIES:
        if sid not in board:
            continue
        listed.add(tag)
        adj, raw = float(board[sid]["adjusted_error"]), float(board[sid]["raw_error"])
        assert abs(adj / values[tag]["adj"] - 1) < 0.02
        assert abs(raw / values[tag]["raw"] - 1) < 0.02
    assert listed == {"us", "artyom", "nurlan", "kai", "aldo", "ddteam"}
    return values
