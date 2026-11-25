# K4: diagnoza, pierwsze wyniki na symulatorze

Wyniki z `scripts/diagnosis_study.py`. To test poprawności i pierwsza liczba do porównań, nie dowód
skuteczności na prawdziwej maszynie.

## Protokół (ustalony przed oceną)

- Dane: symulator, 10 niezależnych maszyn na zbiór, po 6 cykli do awarii (łącznie 60 awarii),
  próbkowanie 60 s, dane „brudne" (braki, skoki, zablokowany czujnik). Czujnik obciążenia
  (`load_pct`) jest wyłączony z diagnozy.
- Zbiór deweloperski: ziarna 1–10 (tu powstały metody i testy). Zbiór testowy: ziarna 101–110,
  uruchomiony raz, po zakończeniu prac nad metodami.
- Założenie: maszyna jest zdrowa na początku każdego ciągłego segmentu danych (po przerwie, np.
  naprawie). Pierwsze 25% okien segmentu to wzorzec odniesienia.
- Alarm to co najmniej 3 kolejne okna powyżej progu. Próg = kwantyl 0,995 wyników z okien
  odniesienia. Czas wykrycia = koniec okna kończącego serię alarmów (bez użycia późniejszych danych).
- Alarm jest trafny, jeśli pojawił się nie wcześniej niż 72 h przed najbliższą następną awarią.
  Inaczej to fałszywy alarm (także zbyt wczesny). Fałszywe alarmy liczone na 100 h monitorowanej pracy.

## Wyniki (próg 72 h)

| Zbiór | Metoda | Wykryte awarie | Recall | Mediana wyprzedzenia | Fałszywe alarmy / 100 h |
| --- | --- | --- | --- | --- | --- |
| deweloperski | reguła progowa (robust z-score) | 25/60 | 0,42 | 44,0 h | 1,21 |
| deweloperski | Isolation Forest | 29/60 | 0,48 | 43,0 h | 0,37 |
| **testowy** | reguła progowa (robust z-score) | 37/60 | 0,62 | 38,3 h | 0,86 |
| **testowy** | Isolation Forest | 31/60 | 0,52 | 41,8 h | 0,38 |

Czułość na próg czasu (zbiór testowy, recall): 24 h → 0,50 (reguła) i 0,28 (IF); 48 h → 0,52 i 0,40;
72 h → 0,62 i 0,52.

## Wnioski i zastrzeżenia

- **Wzorzec odniesienia podgląda przyszłość.** Domyślnie to pierwsze 25% okien segmentu, więc jego
  wielkość zależy od długości całego segmentu, której przy bieżącej pracy maszyny nie znamy. Dla
  analizy zapisanych danych to w porządku, ale czasy wyprzedzenia z tej tabeli są nieco optymistyczne.
  Opcja `reference_windows` (stała liczba okien, jak w K5) jest dostępna w `diagnose`, lecz wyniki
  poniżej policzono bez niej. Powtórzenie badania z opcją stałą, na nowych ziarnach, jest do zrobienia.
- Żadna metoda nie wygrywa jednoznacznie. Isolation Forest daje ok. 2–3 razy mniej fałszywych alarmów,
  reguła progowa wykrywa więcej awarii. Na zbiorze deweloperskim IF był lepszy pod obydwoma
  względami, na testowym nie, więc różnice w recallu mieszczą się w zmienności między zbiorami.
- Próbka jest mała (60 awarii na zbiór) i bez przedziałów ufności. Nie wolno z niej wyciągać
  twierdzeń o wyższości metody.
- Isolation Forest słabo reaguje na przesunięcie pojedynczej cechy jednego czujnika (ocenia punkt w
  całej przestrzeni cech), co pokazał test jednostkowy. Reguła progowa widzi taką zmianę od razu.
- Do sprawdzenia dalej: wyniki osobno dla każdego trybu awarii, wrażliwość na `quantile`, `min_run`
  i długość okna, kompensacja obciążenia zamiast wyłączania czujnika `load_pct`, dane publiczne.
