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
- Ostrzeżenia: poniżej 3 cykli awarii prognoza jest oznaczona jako niewiarygodna, poniżej 10 jako
  orientacyjna, a wiek maszyny dłuższy niż każdy cykl uczący jako ekstrapolacja. Model nie uczy się
  na mniej niż 1 cyklu (baza) lub 2 cyklach (gradient boosting).

## Protokół

- Dane: symulator, 10 maszyn na zbiór po 6 cykli (60 cykli), próbkowanie 60 s, dane „brudne",
  czujnik `load_pct` wyłączony.
- Ocena: leave-one-cycle-out w obrębie maszyny (każdy cykl po kolei jako testowy), wyniki zebrane z
  wszystkich maszyn. Zbiór deweloperski: ziarna 1–10. Zbiór testowy: ziarna 101–110, uruchomiony raz,
  po zakończeniu prac nad metodami.
- Metryki: MAE mediany (ogółem i dla prawdziwego RUL do 72 h i 24 h), pokrycie przedziału 80%
  (cel 0,80) i jego szerokość, wskaźnik Briera dla awarii w ciągu 72 h oraz jego „skill" względem
  stałego przewidywania częstości bazowej (1 = idealnie, 0 = bez poprawy).

## Wyniki (horyzont 72 h)

| Zbiór | Metoda | MAE | MAE (RUL ≤ 72 h) | MAE (RUL ≤ 24 h) | Pokrycie 80% | Szerokość | Brier skill |
| --- | --- | --- | --- | --- | --- | --- | --- |
| deweloperski | age_baseline | 33,4 h | 27,1 h | 20,9 h | 0,52 | 67 h | 0,42 |
| deweloperski | quantile_gbm | 34,1 h | 29,2 h | 26,4 h | 0,39 | 40 h | 0,35 |
| **testowy** | age_baseline | 38,7 h | 34,0 h | 27,1 h | 0,52 | 71 h | 0,39 |
| **testowy** | quantile_gbm | 34,7 h | 29,0 h | 25,3 h | 0,39 | 46 h | 0,40 |

## Wnioski i zastrzeżenia

- Czujniki nie dają wyraźnej przewagi nad samym wiekiem. Na zbiorze deweloperskim gradient boosting
  jest nieco gorszy, na testowym nieco lepszy (MAE 34,7 h wobec 38,7 h), a skill Briera jest
  praktycznie równy. Różnice mieszczą się w zmienności między zbiorami, a próbka to 60 cykli bez
  przedziałów ufności.
- Błędy są duże: mediana myli się średnio o ok. 25–30 h nawet dla okien tuż przed awarią (RUL ≤ 24 h).
  Na tych danych i przy tych cechach RUL nie da się wyznaczyć z dokładnością przydatną do
  planowania co do godziny. Do planowania „w ciągu kilku dni" może wystarczać (skill Briera ok. 0,4).
- Przedziały są za wąskie: pokrycie 0,39 (gradient boosting) i 0,52 (baza) zamiast 0,80. Przy kilku
  cyklach uczących i silnie skorelowanych oknach modele są zbyt pewne siebie. Do wiarygodnych
  przedziałów potrzebna jest kalibracja (np. konforemna), której nie ma.
- Ocena leave-one-cycle-out uczy się także na cyklach zarejestrowanych po cyklu testowym, więc mierzy
  uogólnianie między cyklami, a nie ścisły test „do przodu". Ścisłą wersję dla jednego momentu daje
  `diagsys predict --at <czas>` (model widzi tylko dane i awarie do tej chwili).
- Symulator ma stałą rodzinę krzywych degradacji i niezależne cykle. Wynik na prawdziwej maszynie,
  z małą liczbą awarii i zmiennymi warunkami, będzie gorszy.

## Co dalej (do sprawdzenia)

1. Kalibracja przedziałów (konforemna regresja kwantylowa) i ponowna ocena na **nowych** ziarnach
   (201–210), bo ziarna 101–110 zostały już użyte.
2. Wskaźnik zdrowia z kompensacją obciążenia zamiast wyłączania `load_pct`.
3. Model przeżycia (Cox, lifelines) jako trzeci punkt odniesienia, obsługujący dane cenzurowane.
4. Zbiór publiczny (np. NASA C-MAPSS) obok symulatora.
