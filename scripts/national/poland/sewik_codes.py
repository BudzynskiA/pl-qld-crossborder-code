"""
SEWIK participant codes used by the Polish event-level builders.

Values are the canonical (snake_case) labels of the merged SEWIK participant table
(`uczestnik`) after national preprocessing:

- `poziom_obrazen` (injury level): nieranny, lekko_ranny, ciezko_ranny,
  zmarly_w_ciagu_30_dni, zmarly_na_miejscu
- `czy_sprawca` (participant is the at-fault party): tak, nie, nieznane
- `rodzaj_uczestnika` (participant type): kierujacy, pieszy, pasazer, osoba_uwr, nieznane

Only the constants needed here are defined; the strings are data values and must
match the merged tables exactly.
"""
from __future__ import annotations

COL_POZIOM = "poziom_obrazen"

POZIOM_NIERANNY = "nieranny"  # not injured
POZIOM_LEKKO = "lekko_ranny"  # slightly injured
POZIOM_CIEZKO = "ciezko_ranny"  # seriously injured
POZIOM_ZM_30 = "zmarly_w_ciagu_30_dni"  # died within 30 days
POZIOM_ZM_MIEJSCE = "zmarly_na_miejscu"  # died at the scene

SPRAWCA_TAK = "tak"  # at-fault participant

RODZ_PIESZY = "pieszy"  # pedestrian
