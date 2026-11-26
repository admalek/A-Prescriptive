# K4: diagnoza, pierwsze wyniki na symulatorze

Wyniki z `scripts/diagnosis_study.py`. To test poprawności i pierwsza liczba do porównań, nie dowód
skuteczności na prawdziwej maszynie.

## Protokół (ustalony przed oceną)

- Dane: symulator, 10 niezależnych maszyn na zbiór, po 6 cykli do awarii (łącznie 60 awarii),
  próbkowanie 60 s, dane „brudne" (braki, skoki, zablokowany czujnik). Czujnik obciążenia
  (`load_pct`) jest wyłączony z diagnozy.
- Zbiór deweloperski: ziarna 1–10 (tu powstały metody i testy). Zbiór testowy: ziarna 201–210,
  uruchomiony raz, po ostatniej zmianie metod. Ziarna 101–110 posłużyły do pierwszej wersji tej
  oceny i nie są już zbiorem testowym.
- Założenie: maszyna jest zdrowa na początku każdego ciągłego segmentu danych (po przerwie, np.
  naprawie). Wzorzec odniesienia to pierwsze 144 okna segmentu (24 h przy kroku 10 min), stała
  liczba, więc ocena okna nie zależy od przyszłości.
- Alarm to co najmniej 3 kolejne okna powyżej progu. Próg = kwantyl 0,995 wyników z okien
  odniesienia. Czas wykrycia = koniec okna kończącego serię alarmów (bez użycia późniejszych danych).
- Alarm jest trafny, jeśli pojawił się nie wcześniej niż 72 h przed najbliższą następną awarią.
  Inaczej to fałszywy alarm (także zbyt wczesny). Fałszywe alarmy liczone na 100 h monitorowanej pracy.

## Wyniki (próg 72 h, stały wzorzec 144 okien)

| Zbiór | Metoda | Wykryte awarie | Recall | Mediana wyprzedzenia | Fałszywe alarmy / 100 h |
| --- | --- | --- | --- | --- | --- |
| deweloperski (1–10) | reguła progowa (robust z-score) | 29/60 | 0,48 | 52,1 h | 1,09 |
| deweloperski (1–10) | Isolation Forest | 26/60 | 0,43 | 42,3 h | 0,31 |
| **testowy (201–210)** | reguła progowa (robust z-score) | 34/60 | 0,57 | 36,3 h | 0,90 |
| **testowy (201–210)** | Isolation Forest | 23/60 | 0,38 | 40,6 h | 0,54 |

Czułość na próg czasu (zbiór testowy, recall): 24 h → 0,40 (reguła) i 0,22 (IF); 48 h → 0,42 i 0,38;
72 h → 0,57 i 0,38.

Porównanie na zbiorze deweloperskim ze starym wzorcem (25% segmentu): reguła 0,42 / IF 0,48, czyli
stały wzorzec nie pogorszył wyników (reguła lepsza, IF nieco gorszy, w granicach zmienności).

## Wnioski i zastrzeżenia

- **Wzorzec odniesienia jest już stały** (poprawka względem pierwszej wersji, w której podglądał
  przyszłość). Liczby powyżej są więc możliwe do uzyskania w czasie rzeczywistym. Tryb ułamkowy
  zostaje jako opcja (`--reference-windows 0`) do analizy zapisanych danych.
- Żadna metoda nie wygrywa jednoznacznie. Na nowych ziarnach reguła progowa wykrywa więcej awarii (0,57 vs 0,38),
  a Isolation Forest daje mniej fałszywych alarmów (0,54 vs 0,90 na 100 h). Na zbiorze
  deweloperskim różnica w recallu była mała, więc wynik jest wrażliwy na zbiór.
- Próbka jest mała (60 awarii na zbiór) i bez przedziałów ufności. Nie wolno z niej wyciągać
  twierdzeń o wyższości metody.
- Isolation Forest słabo reaguje na przesunięcie pojedynczej cechy jednego czujnika (ocenia punkt w
  całej przestrzeni cech), co pokazał test jednostkowy. Reguła progowa widzi taką zmianę od razu.
- Do sprawdzenia dalej: wyniki osobno dla każdego trybu awarii, wrażliwość na `quantile`, `min_run`
  i długość okna, kompensacja obciążenia zamiast wyłączania czujnika `load_pct`, dane publiczne.
