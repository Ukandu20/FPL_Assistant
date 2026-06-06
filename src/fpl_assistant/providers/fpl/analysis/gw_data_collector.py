import os
import sys
import csv

# ---------- path helpers -----------------------------------------------------

def _find_one(candidates):
    """Return first existing path from candidates or None."""
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None

def _season_candidates(root, filename):
    """
    Try:
      <root>/<filename>
      <root>/season/<filename>
      <root>/../<filename>
      <root>/../season/<filename>
    """
    parent = os.path.abspath(os.path.join(root, os.pardir))
    return [
        os.path.join(root, filename),
        os.path.join(root, "season", filename),
        os.path.join(parent, filename),
        os.path.join(parent, "season", filename),
    ]

def _open_csv_any(candidates, mode="r", encoding="utf-8"):
    path = _find_one(candidates)
    if path is None:
        raise FileNotFoundError(f"Not found. Tried: {candidates}")
    return open(path, mode, encoding=encoding)

def _coerce_bool(x):
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    return s in {"1", "true", "t", "yes", "y"}

# ---------- loaders ----------------------------------------------------------

def get_teams(directory):
    """
    teams.csv is written under <SEASON>/season/ now.
    Expect columns: id, name  (FPL bootstrap teams export)
    """
    fin = _open_csv_any(_season_candidates(directory, "teams.csv"))
    reader = csv.DictReader(fin)
    teams = {}
    for row in reader:
        # tolerate either 'id' or 'code' if your export differs
        tid = row.get("id") or row.get("code")
        name = row.get("name") or row.get("short_name") or row.get("team_name")
        if tid is None or name is None:
            continue
        teams[int(tid)] = name
    return teams

def get_fixtures(directory):
    """
    fixtures.csv is under <SEASON>/season/.
    Expect columns: id, team_h, team_a
    """
    fin = _open_csv_any(_season_candidates(directory, "fixtures.csv"))
    reader = csv.DictReader(fin)
    fixtures_home, fixtures_away = {}, {}
    for row in reader:
        fid = int(row["id"])
        fixtures_home[fid] = int(row["team_h"])
        fixtures_away[fid] = int(row["team_a"])
    return fixtures_home, fixtures_away

def get_positions(directory):
    """
    players_raw.csv is produced at <SEASON>/ (parent of season/).
    Expect columns: id, element_type, first_name, second_name
    """
    fin = _open_csv_any(_season_candidates(directory, "players_raw.csv"))
    reader = csv.DictReader(fin)
    pos_dict = {"1": "GK", "2": "DEF", "3": "MID", "4": "FWD", "5": "AM"}
    positions, names = {}, {}
    for row in reader:
        pid = int(row["id"])
        et = str(row["element_type"])
        positions[pid] = pos_dict.get(et, et)
        names[pid] = f"{row.get('first_name','').strip()} {row.get('second_name','').strip()}".strip()
    return names, positions

def get_expected_points(gw, directory):
    """
    xP<gw>.csv lives in gws/ (this is passed in as output_dir).
    """
    xPoints = {}
    try:
        fin = open(os.path.join(directory, f"xP{gw}.csv"), "r", encoding="utf-8")
        reader = csv.DictReader(fin)
        for row in reader:
            try:
                xPoints[int(row["id"])] = float(row.get("xP", 0.0))
            except Exception:
                continue
    except Exception:
        return xPoints
    return xPoints

# ---------- merge & collect --------------------------------------------------

def merge_gw(gw, gw_directory):
    merged_gw_filename = "merged_gw.csv"
    gw_filename = f"gw{gw}.csv"
    gw_path = os.path.join(gw_directory, gw_filename)
    fin = open(gw_path, "r", encoding="utf-8")
    reader = csv.DictReader(fin)
    fieldnames = list(reader.fieldnames) + ["GW"]
    rows = []
    for row in reader:
        row["GW"] = gw
        rows.append(row)
    out_path = os.path.join(gw_directory, merged_gw_filename)
    # append; write header only for gw==1
    fout = open(out_path, "a", encoding="utf-8", newline="")
    writer = csv.DictWriter(fout, fieldnames=fieldnames, lineterminator="\n")
    print(gw)
    if gw == 1:
        writer.writeheader()
    for row in rows:
        writer.writerow(row)

def collect_gw(gw, directory_name, output_dir, root_directory_name="data/2024-25"):
    """
    directory_name: path to <SEASON>/players/ (contains per-player folders with gw.csv)
    output_dir:     path to <SEASON>/gws/
    root_directory_name: path used to find fixtures/teams/players_raw (now season-aware)
    """
    fixtures_home, fixtures_away = get_fixtures(root_directory_name)
    teams = get_teams(root_directory_name)
    names, positions = get_positions(root_directory_name)
    xPoints = get_expected_points(gw, output_dir)

    rows = []
    # Walk players dir robustly (absolute or relative)
    for root, _, files in os.walk(directory_name):
        for fname in files:
            if fname == "gw.csv":
                fpath = os.path.join(root, fname)
                fin = open(fpath, "r", encoding="utf-8")
                reader = csv.DictReader(fin)
                # capture once (robust to missing) — we’ll rebuild final header later
                fieldnames = reader.fieldnames or []
                for row in reader:
                    try:
                        # FBref/FPL dumps sometimes have 'round' as floaty string; coerce safely
                        if int(float(row["round"])) != gw:
                            continue
                        pid = int(os.path.basename(root).split("_")[-1])
                        name = names.get(pid, str(pid))
                        position = positions.get(pid, "")
                        fixture = int(row["fixture"])
                        was_home = _coerce_bool(row.get("was_home", False))
                        row["team"] = teams[(fixtures_home if was_home else fixtures_away)[fixture]]
                        row["name"] = name
                        row["position"] = position
                        row["xP"] = xPoints.get(pid, 0.0)
                        rows.append(row)
                    except Exception:
                        # Skip malformed rows silently to keep pipeline resilient
                        continue

    # Final header: stable prefix + original columns (dedup preserving order)
    base = ["name", "position", "team", "xP"]
    # If rows is empty, still write header using a common guess
    tail = list(dict.fromkeys(rows[0].keys() if rows else ["element", "opponent_team", "total_points", "round", "fixture", "was_home"]))
    # Ensure base appears first and not duplicated
    tail = [c for c in tail if c not in base]
    fieldnames_final = base + tail

    outf = open(os.path.join(output_dir, f"gw{gw}.csv"), "w", encoding="utf-8", newline="")
    writer = csv.DictWriter(outf, fieldnames=fieldnames_final, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

def collect_all_gws(directory_name, output_dir, root_dir):
    for i in range(1, 17):
        collect_gw(i, directory_name, output_dir, root_dir)

def merge_all_gws(num_gws, gw_directory):
    for i in range(1, num_gws):
        merge_gw(i, gw_directory)

def main():
    # collect_all_gws(sys.argv[1], sys.argv[2], sys.argv[3])
    merge_all_gws(int(sys.argv[1]), sys.argv[2])
    # collect_gw(22, sys.argv[1], sys.argv[2])

if __name__ == '__main__':
    main()
