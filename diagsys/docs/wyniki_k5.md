# K5: predykcja RUL, pierwsze wyniki na symulatorze

Wyniki z `scripts/prediction_study.py`. To test poprawności i punkt odniesienia do dalszych
porównań, nie dowód skuteczności na prawdziwej maszynie.

## Co jest przewidywane i jak

- Cel: czas od końca okna cech do najbliższej następnej zapisanej awarii (RUL), jako kwantyle
  10/25/50/75/90%. Kontrakt `Forecast` niesie medianę, przedział 80% i przybliżone prawdopodobieństwo
  awarii w horyzoncie (kawałkami liniowa dystrybuanta z pięciu kwantyli, niekalibrowana poza oceną
  Briera).
- Bez wglądu w przyszłość: cechy, wiek od ostatniej naprawy i wzorzec odniesienia (stałe 144 okna,
  czyli 24 h na początku segmentu) są znane w chwili prognozy. Okna z wzorca nie są prognozowane.
  Test jednostkowy sprawdza, że przycięcie danych w czasie nie zmienia cech wcześniejszych okien.
- Cenzurowanie: okna bez późniejszej awarii nie służą do uczenia ani oceny.
- Metody na wspólnym zbiorze i w tym samym formacie:
  - `age_baseline`: punkt odniesienia bez czujników, kwantyle długości życia cykli uczących minus wiek
  - `quantile_gbm`: gradient boosting z funkcją kwantylową na cechach zdrowia (robust z-score każdego
    czujnika względem wzorca, jego średnie kroczące z 1, 6 i 24 h, wiek). Hiperparametry stałe.
  - `quantile_gbm_cal`: ten sam model z przedziałem 80% poszerzonym o korektę konforemną (CQR,
    Romano i in. 2019) w skali log(1+RUL). Korekta to kwantyl rzędu (1−α)(1+1/n), α = 0,2, miar
    chybienia `max(q10 − y, y − q90)` liczonych na cyklach uczących metodą wewnętrznego
    leave-one-cycle-out. Dolne kwantyle przesuwa w dół, górne w górę o tę samą wartość, mediana
    pozostaje bez zmian. Wymaga co najmniej 3 cykli.
- Ostrzeżenia: poniżej 3 cykli awarii prognoza jest oznaczona jako niewiarygodna, poniżej 10 jako
  orientacyjna, a wiek maszyny dłuższy niż każdy cykl uczący jako ekstrapolacja. Model nie uczy się
  na mniej niż 1 cyklu (baza) lub 2 cyklach (gradient boosting).

## Protokół

- Dane: symulator, 10 maszyn na zbiór po 6 cykli (60 cykli), próbkowanie 60 s, dane „brudne",
  czujnik `load_pct` wyłączony.
- Ocena: leave-one-cycle-out w obrębie maszyny (każdy cykl po kolei jako testowy), wyniki zebrane z
  wszystkich maszyn. Zbiór deweloperski: ziarna 1–10. Zbiór testowy: ziarna 201–210, uruchomiony raz,
  po zakończeniu prac nad metodami (ziarna 101–110 były testowe w pierwszej wersji tej oceny
  i nie są już używane).
- Metryki: MAE mediany (ogółem i dla prawdziwego RUL do 72 h i 24 h), pokrycie przedziału 80%
  (cel 0,80) i jego szerokość, wskaźnik Briera dla awarii w ciągu 72 h oraz jego „skill" względem
  stałego przewidywania częstości bazowej (1 = idealnie, 0 = bez poprawy).

## Wyniki (horyzont 72 h)

| Zbiór | Metoda | MAE | MAE (RUL ≤ 72 h) | MAE (RUL ≤ 24 h) | Pokrycie 80% | Pokrycie (RUL ≤ 72 h) | Szerokość | Brier skill |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| deweloperski (1–10) | age_baseline | 33,4 h | 27,1 h | 20,9 h | 0,52 | 0,53 | 67 h | 0,42 |
| deweloperski (1–10) | quantile_gbm | 34,1 h | 29,2 h | 26,4 h | 0,39 | 0,46 | 40 h | 0,35 |
| deweloperski (1–10) | quantile_gbm_cal | 34,4 h | 28,1 h | 24,9 h | 0,79 | 0,69 | 125 h | 0,43 |
| **testowy (201–210)** | age_baseline | 42,7 h | 36,0 h | 28,8 h | 0,45 | 0,45 | 78 h | 0,29 |
| **testowy (201–210)** | quantile_gbm | 34,7 h | 24,0 h | 23,3 h | 0,38 | 0,45 | 49 h | 0,41 |
| **testowy (201–210)** | quantile_gbm_cal | 35,4 h | 22,9 h | 21,7 h | 0,81 | 0,72 | 146 h | 0,44 |

## Wnioski i zastrzeżenia

- **Kalibracja działa tam, gdzie miała działać.** Pokrycie przedziału 80% wzrosło z 0,38–0,39 do
  0,79 (dev) i 0,81 (test), czyli do celu 0,80, bez zmiany mediany (MAE prawie bez zmian). Cena:
  przedział jest ok. 3 razy szerszy (146 h na zbiorze testowym), więc mówi „kilka dni", nie
  „godzina". Skill Briera lekko wzrósł (0,41 → 0,44 na teście), co sugeruje, że i prawdopodobieństwo
  awarii wyznaczone z kwantyli jest lepsze.
- Pokrycie dla okien tuż przed awarią (RUL ≤ 72 h) jest nadal poniżej celu (0,69 i 0,72): korekta jest
  jedna dla wszystkich okien, a model myli się bardziej blisko końca życia. Nie należy odczytywać
  przedziału jako gwarancji dla samych końcowych godzin.
- Gwarancja konforemna jest tylko przybliżona: okna jednego cyklu są silnie skorelowane, a cykli
  uczących jest kilka. Wynik potwierdza dopiero ocena na nowych maszynach (tu: ziarna 201–210).
- Czujniki dają wyraźną przewagę nad samym wiekiem dopiero na zbiorze testowym (MAE 34,7 h wobec
  42,7 h, skill 0,41 wobec 0,29). Na deweloperskim różnica była odwrotna (33,4 h wobec 34,1 h). To
  różnica między zbiorami, a nie ustalona przewaga; próbka to 60 cykli bez przedziałów ufności.
- Błędy mediany są duże: ok. 22–24 h nawet dla okien tuż przed awarią (RUL ≤ 24 h). RUL nie da się
  tu wyznaczyć z dokładnością do godzin; do planowania „w ciągu kilku dni" może wystarczać.
- Ocena leave-one-cycle-out uczy się także na cyklach zarejestrowanych po cyklu testowym, więc mierzy
  uogólnianie między cyklami, a nie ścisły test „do przodu". Ścisłą wersję dla jednego momentu daje
  `diagsys predict --at <czas>` (model widzi tylko dane i awarie do tej chwili).
- Symulator ma stałą rodzinę krzywych degradacji i niezależne cykle. Wynik na prawdziwej maszynie,
  z małą liczbą awarii i zmiennymi warunkami, będzie gorszy.

## Co dalej (do sprawdzenia)

1. Kalibracja zależna od wieku lub stanu (przedział węższy daleko od awarii, szerszy tuż przed nią),
   bo jedna korekta dla wszystkich okien daje niedopokrycie końcówki cyklu. Następna ocena końcowa
   wymaga nowych ziaren (211–220), bo 201–210 zostały użyte.
2. Wskaźnik zdrowia z kompensacją obciążenia zamiast wyłączania `load_pct`.
3. Model przeżycia (Cox, lifelines) jako trzeci punkt odniesienia, obsługujący dane cenzurowane.
4. Zbiór publiczny (np. NASA C-MAPSS) obok symulatora.
