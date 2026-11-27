# K7: ocena polityk serwisowych na symulatorze

Wyniki z `scripts/prescription_study.py`. To test, czy działanie na podstawie prognozy opłaca się
bardziej niż proste reguły, na danych symulowanych. Nie jest dowodem dla prawdziwej maszyny.

## Protokół

- Dane i prognozy jak w K5: symulator, 10 maszyn na zbiór, po 6 cykli do awarii (60 cykli), dane
  „brudne", bez `load_pct`. Prognozy to prognozy „poza próbą" z walidacji leave-one-cycle-out, więc
  polityka nie widzi własnej awarii.
- Zbiór deweloperski: ziarna 1–10. Zbiór testowy: ziarna 201–210, uruchomiony raz, po dodaniu
  wszystkich polityk (wariant łączony dodałem po obejrzeniu wyników deweloperskich pierwszych
  polityk, więc jest wybrany na zbiorze deweloperskim; progi 0,6 i 0,8 dla wieku ustalone z góry).
- Czas na zorganizowanie serwisu: 24 h. Koszty względne: planowy serwis = 1, awaria = 3, 5 lub 10
  (założenie, nie dane). Cykl kończy się awarią (koszt = ratio) albo serwisem, który zaczął się
  przed awarią (koszt = 1).
- Polityki: `run_to_failure` (bez serwisu), `fixed_age(f)` (serwis po f × mediana życia
  pozostałych cykli maszyny), `quantile_gbm_cal(r)` (reguła ryzyka z K7: zamów serwis, gdy
  prognozowane prawdopodobieństwo awarii w czasie organizacji serwisu ≥ r), `hybrid(0,8; r)`
  (wcześniejszy z serwisu po wieku i z serwisu z prognozy), `oracle` (serwis w chwili awarii:
  nieosiągalny, pokazuje zapas).

## Wyniki (awaria = 5 × serwis; r = 0,1)

| Zbiór | Polityka | Cykle zakończone awarią | Koszt na cykl | Niewykorzystane życie | Koszt na 1000 h |
| --- | --- | --- | --- | --- | --- |
| dev (1–10) | bez serwisu | 100% | 5,00 | – | 22,1 |
| dev (1–10) | stały wiek 0,6 | 0% | 1,00 | 39% | 7,50 |
| dev (1–10) | stały wiek 0,8 | 10% | 1,40 | 23% | 7,99 |
| dev (1–10) | prognoza (r = 0,1) | 30% | 2,20 | 16% | 11,0 |
| dev (1–10) | łączona (0,8; r = 0,1) | 7% | 1,27 | 23% | 7,31 |
| **test (201–210)** | bez serwisu | 100% | 5,00 | – | 22,5 |
| **test (201–210)** | stały wiek 0,6 | 0% | 1,00 | 37% | 7,49 |
| **test (201–210)** | stały wiek 0,8 | 23% | 1,93 | 26% | 11,2 |
| **test (201–210)** | prognoza (r = 0,1) | 32% | 2,27 | 15% | 11,4 |
| **test (201–210)** | łączona (0,8; r = 0,1) | 13% | 1,53 | 25% | 9,10 |
| **test (201–210)** | oracle | 0% | 1,00 | 0% | 4,49 |

Dla awarii 3 × i 10 × droższej kolejność na zbiorze testowym jest ta sama (koszt na 1000 h: stały
wiek 0,6: 7,49 w obu przypadkach; łączona: 7,51 i 13,05; prognoza: 8,24 i 19,42).

## Wnioski i zastrzeżenia

- **Serwis oparty wyłącznie na prognozie przegrywa na tym symulatorze z prostą regułą wieku.**
  Co trzeci cykl kończy się awarią, bo prognoza przy końcu życia jest zbyt optymistyczna (to samo
  widać w pokryciu przedziału dla RUL ≤ 72 h w `wyniki_k5.md`), a 24 h na organizację serwisu to
  mniej niż typowy błąd mediany (ok. 22–35 h).
- **Stały wiek 0,6 × typowe życie nie ma awarii i najniższy koszt**, ale marnuje 37–39% życia.
  Działa tak dobrze, bo cykle w symulatorze różnią się długością umiarkowanie, więc sam wiek jest
  bardzo informatywny. Przy maszynach o bardziej zmiennym życiu przewaga reguły wieku się zmniejszy;
  tego tu nie sprawdzono.
- Wariant łączony (wiek 0,8 + sygnał z prognozy) zmniejsza liczbę awarii względem samego wieku 0,8
  (13% wobec 23% na teście, 7% wobec 10% na dev) i jest od niego tańszy, więc prognoza wnosi
  wartość jako ostrzeżenie przed końcem terminu wieku. Nie pokonuje jednak wieku 0,6.
- Nie wynika stąd, że prognoza jest zbędna: daje przedział, ostrzeżenia i diagnozę przyczyn, a
  symulator ma jedną rodzinę krzywych degradacji. Wynika, że **obietnicy „prognoza wygrywa z
  prostą regułą" nie wolno składać bez oceny na własnych danych**, i to jest wynik do pracy.
- Koszty są względne i założone. Zbiór to 60 cykli bez przedziałów ufności; różnice rzędu kilku
  punktów procentowych awarii nie są rozstrzygające.
- Narzędzie `prescribe` rekomenduje termin z reguły ryzyka i (gdy znane) kosztu. Ten eksperyment
  ocenia samą regułę ryzyka. Nie ocenia reguły kosztu ani części tekstowej (wskazówki z
  dokumentacji).

## Co dalej

1. Zmienna długość życia w symulatorze (różne obciążenia, różne tryby awarii), żeby sprawdzić, kiedy
   wiek przestaje wystarczać.
2. Kalibracja prognozy zależna od stanu (patrz `wyniki_k5.md`), bo to ona decyduje o spóźnionych
   serwisach.
3. Dane rzeczywiste lub publiczne z dłuższą historią awarii.
