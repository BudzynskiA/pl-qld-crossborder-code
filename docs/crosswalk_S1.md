# Supplementary Table S1. Harmonization crosswalk (Poland SEWIK and Queensland TMR)

**citekey:** `budzynski_crossborder_pl_qld`  
**Source scripts:** `scripts/harmonization/`  
**Audit:** `results/harmonized_flags_country_leakage_audit.csv` (27 indicators: 19 OK, 1 WARN, 7 FAIL; plus 1 location group check)

| Domain | Harmonized indicator (paper label) | Polish source field | Polish code / rule | Queensland source field | Queensland code / rule | Modelling | Exclusion reason |
|--------|------------------------------|---------------------|--------------------|-------------------------|------------------------|-----------|------------------|
| Outcome | Binary fatal-crash indicator | `ilosc_zmarlych` | death count > 0 | `Count_Casualty_Fatality` | fatality count > 0 | Outcome | - |
| Weather | Clear or cloudy without precipitation | `warunki_atmosferyczne` | tokens `dobre` or `pochmurno`, no adverse weather | `Crash_Atmospheric_Condition` | `Clear` | Include | - |
| Weather | Rain | `warunki_atmosferyczne` | token `opady_deszczu` | `Crash_Atmospheric_Condition` | `Raining` | Include | - |
| Weather | Fog or smoke | `warunki_atmosferyczne` | token `mgla_dym` | `Crash_Atmospheric_Condition` | `Fog` | Include | - |
| Weather | Other or unknown weather | `warunki_atmosferyczne` | tokens `oslepiajace_slonce`, `nieznane`, empty, or unmapped | `Crash_Atmospheric_Condition` | `Smoke/Dust` or unmapped | Include | - |
| Weather | Snow or hail | `warunki_atmosferyczne` | token `opady_sniegu_gradu` | - | no TMR code (always 0) | Exclude | Active only in Poland |
| Weather | Strong wind | `warunki_atmosferyczne` | token `silny_wiatr` | - | no TMR code (always 0) | Exclude | Active only in Poland |
| Crash type | Head-on | `rodzaj_zdarzenia` | `zderzenie_czolowe` | `Crash_Nature` | `Head-on` | Include | - |
| Crash type | Side | `rodzaj_zdarzenia` | `zderzenie_boczne` | `Crash_Nature` | `Angle`, `Sideswipe` | Include | - |
| Crash type | Rear-end | `rodzaj_zdarzenia` | `zderzenie_tylne` | `Crash_Nature` | `Rear-end` | Include | - |
| Crash type | Pedestrian | `rodzaj_zdarzenia` | `najechanie_na_pieszego` | `Crash_Nature` / `Crash_Type` | `Hit pedestrian` | Include | - |
| Crash type | Struck stationary vehicle | `rodzaj_zdarzenia` | `najechanie_na_pojazd_unieruchomiony` | `Crash_Nature` | `Hit parked vehicle` | Include | - |
| Crash type | Struck fixed object | `rodzaj_zdarzenia` | `najechanie_na_drzewo`, `najechanie_na_slup_znak`, `najechanie_na_bariere`, `najechanie_na_zwierze`, `najechanie_na_dziure_wyboi_garb` | `Crash_Nature` | `Hit object`, `Hit animal` | Include | - |
| Crash type | Overturn | `rodzaj_zdarzenia` | `wywrocenie_pojazdu` | `Crash_Nature` | `Overturned` | Include | - |
| Crash type | Passenger fall | `rodzaj_zdarzenia` | `zdarzenie_z_pasazerem` | `Crash_Nature` | `Fall from vehicle` | Include | - |
| Crash type | Other crash type | `rodzaj_zdarzenia` | `zdarzenie_z_osoba_uwr`, `inne`, unmapped | `Crash_Nature` / `Crash_Type` | `Struck by external/internal load`, `Other`, unmapped | Include (retained with warning) | Strong frequency asymmetry (PL 4.2% vs QLD 0.13%) |
| Crash type | Level crossing | `rodzaj_zdarzenia` | `najechanie_na_zapore_kolejowa` | - | no TMR code (always 0) | Exclude | Active only in Poland |
| Pavement | Hard or paved surface | `nawierzchnia_drogi` | `twarda` | `Crash_Road_Surface_Condition` | starts with `Sealed` | Include | - |
| Pavement | Unpaved surface | `nawierzchnia_drogi` | `gruntowa` | `Crash_Road_Surface_Condition` | starts with `Unsealed` | Include | - |
| Pavement | Dry surface | `stan_nawierzchni` | token `sucha`, or no wet, slippery, ice or snow token (missing condition treated as dry) | `Crash_Road_Surface_Condition` | ends with `Dry` | Include | - |
| Pavement | Wet surface | `stan_nawierzchni` | tokens `mokra`, `kaluze_rozlewiska` | `Crash_Road_Surface_Condition` | ends with `Wet` | Include | - |
| Pavement | Slippery / contaminated | `stan_nawierzchni` | token `zanieczyszczona` | - | no TMR code (always 0) | Exclude | Active only in Poland |
| Pavement | Ice | `stan_nawierzchni` | token `oblodzenie_zasniezenie` | - | no TMR code (always 0) | Exclude | Active only in Poland |
| Pavement | Snow-covered | `stan_nawierzchni` | token `oblodzona_zasniezona` | - | no TMR code (always 0) | Exclude | Active only in Poland |
| Pavement | Damaged surface | `stan_nawierzchni` | tokens `dziury_wyboje`, `koleiny_garby` | - | no TMR code (always 0) | Exclude | Active only in Poland |
| Location | Motorway (Queensland: motorway or high-speed state road proxy) | `rodzaj_drogi` | `autostrada` | `State_Road_Name`, `Crash_Controlling_Authority`, `Crash_Speed_Limit` | name contains motorway/freeway OR State-controlled + 100–110 km/h | Include | - |
| Location | Urban area (Poland: built-up area; Queensland: ABS Major Cities class) | `obszar` | `obszar_zabudowany` (not motorway/expressway) | `Loc_ABS_Remoteness` | `Major Cities` (and not motorway) | Include | - |
| Location | Non-urban area | `obszar`, `rodzaj_drogi` | `obszar_niezabudowany` or `ekspresowa` (not motorway) | `Loc_ABS_Remoteness` | all other remoteness classes, incl. unknown (not motorway, not major city) | Include | - |
| Lighting (extension) | Daylight | `oswietlenie` | `swiatlo_dzienne` | `Crash_Lighting_Condition` | `Daylight` | Sensitivity analysis only | - |
| Lighting (extension) | Darkness, street lighting | `oswietlenie` | `noc_oswietlona` | `Crash_Lighting_Condition` | `Darkness - Lighted` | Sensitivity analysis only | - |
| Lighting (extension) | Darkness, no lighting | `oswietlenie` | `noc_nieoswietlona` | `Crash_Lighting_Condition` | `Darkness - Not lighted` | Sensitivity analysis only | - |
| Lighting (extension) | Dawn or dusk | `oswietlenie` | `swit_zmrok` | `Crash_Lighting_Condition` | `Dawn/Dusk` | Sensitivity analysis only | - |
| Speed limit (extension) | 50 km/h or less | `predkosc_dopuszczalna` | limit <= 50 | `Crash_Speed_Limit` | `0 - 50 km/h` | Sensitivity analysis only | - |
| Speed limit (extension) | 60 km/h | `predkosc_dopuszczalna` | 50 < limit <= 60 | `Crash_Speed_Limit` | `60 km/h` | Sensitivity analysis only | - |
| Speed limit (extension) | 70 km/h | `predkosc_dopuszczalna` | 60 < limit <= 70 | `Crash_Speed_Limit` | `70 km/h` | Sensitivity analysis only | - |
| Speed limit (extension) | 80 to 90 km/h | `predkosc_dopuszczalna` | 70 < limit <= 90 | `Crash_Speed_Limit` | `80 - 90 km/h` | Sensitivity analysis only | - |
| Speed limit (extension) | 100 km/h or more | `predkosc_dopuszczalna` | limit > 90 (includes 120 and 140 km/h) | `Crash_Speed_Limit` | `100 - 110 km/h` | Sensitivity analysis only | - |

**Note:** Polish atmospheric and pavement fields allow semicolon-separated multi-token activation; Queensland fields are single-coded per sub-domain. Internal harmonized column names (`pogoda_*`, `typ_*`, `naw_*`, `lok_*`) map to the English paper labels in Table 3 of the main manuscript.

**Note on location indicators:** The location indicators are functional analogues. The Queensland open file has no road class field; its motorway indicator is a proxy that covers 17.3% of Queensland crashes (7.6% on roads named motorway or freeway; the remainder mainly high-speed state highways outside major cities), against 1.4% for the Polish motorway road class. The Polish urban indicator follows the legal built-up area designation; the Queensland urban indicator follows the Australian Bureau of Statistics (ABS) Remoteness Area class Major Cities. Alternative definitions are tested in the sensitivity analysis (main manuscript, Table 13).

**Note on extension indicators:** Lighting and speed-limit band indicators are not part of the 20-predictor main analysis; they are used only in the contract-extension sensitivity analysis (29 predictors; main manuscript, Table 13). Each is active in both registers. The speed-limit bands are strongly asymmetric (up to 50 km/h: 70.2% of Polish and 20.1% of Queensland crashes; 60 km/h: 2.4% and 47.6%) because the general speed limit in built-up areas is 50 km/h in Poland and 60 km/h in Queensland.
