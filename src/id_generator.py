# id_generator.py
import hashlib
import unidecode

def normalize(s: str) -> str:
    return unidecode.unidecode(s.strip().lower())

def generate_numeric_id(*args, length: int = 8) -> int:
    base = "_".join(normalize(arg) for arg in args if arg)
    hashed = int(hashlib.md5(base.encode()).hexdigest(), 16)
    return hashed % (10**length)

def generate_player_id(name: str, dob_year: str, position: str, team: str) -> int:
    # Stable ID only derived from name and dob_year; ignore position & team
    return generate_numeric_id(name, dob_year)

def generate_team_id(team_name: str, season: str) -> int:
    # Unique team ID per team-season
    return generate_numeric_id(team_name, season)