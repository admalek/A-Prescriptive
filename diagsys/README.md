# diagsys

Uniwersalny, lokalny system wieloagentowy do diagnostyki, predykcji i preskrypcji dla maszyn.
Działa na laptopie (CPU, 16 GB RAM), offline, na modelach otwartych (Bielik do języka polskiego,
klasyczne ML do liczenia). Najpierw budujemy system ogólny, potem "uczymy" go konkretnej maszyny
przez profil maszyny, jej dokumentację (osobna baza RAG) i jej dane.

## Zasady

1. Model językowy planuje, wyjaśnia i pisze raporty. Liczby liczą narzędzia deterministyczne.
2. System tylko doradza. Nie ma zapisu do maszyn ani sterowników.
3. Każda teza ma źródło (dokument, wynik narzędzia, dane).
4. Moduły rozmawiają przez kontrakty (`src/diagsys/contracts/`) i porty (`src/diagsys/ports.py`),
   więc laptop można później zamienić na serwer bez przepisywania logiki.

## Szybki start

```bash
make setup     # uv sync --extra dev
make test      # testy
make doctor    # sprawdza laptop: RAM, dysk, serwer modeli, pakiety
cp .env.example .env
```

## Jak dodać maszynę (bez zmian w kodzie)

```bash
uv run diagsys machine new press-01 --name "Prasa hydrauliczna" --type press
```

Powstaje `data/machines/press-01/`:

| Katalog / plik | Do czego |
| --- | --- |
| `profile.yaml` | czujniki, tryby awarii, koszty przestoju, nazwa kolekcji RAG (`docs_press-01`) |
| `docs/` | instrukcje i dokumentacja maszyny (indeksowane do RAG od K6) |
| `raw/` | surowe dane (CSV, Parquet) |
| `processed/` | raport jakości, dane oczyszczone i cechy (Parquet), od K3 |
| `models/` | modele wytrenowane na danych tej maszyny |
| `reports/` | wygenerowane raporty |

## Dane: symulator, import, podgląd (K2)

```bash
# 1. dane syntetyczne (do testów systemu, zanim będą prawdziwe dane)
uv run diagsys simulate press-01 --cycles 5 --dirty    # CSV + etykiety awarii w raw/

# 2. import do bazy (DuckDB, jeden plik data/diagsys.duckdb); bezpieczny do powtórzenia
uv run diagsys ingest press-01 data/machines/press-01/raw/sim_seed42.csv
uv run diagsys ingest-events press-01 data/machines/press-01/raw/sim_seed42_failures.csv

# 3. co jest w bazie
uv run diagsys data summary press-01
```

Import prawdziwych plików (CSV, TSV, TXT, Parquet) obsługuje typowe eksporty z fabryk:

| Sytuacja | Opcja |
| --- | --- |
| kolumna na czujnik (domyślnie) lub jeden wiersz na odczyt | `--layout wide` / `--layout long` |
| czas bez strefy, np. `19.11.2026 08:00:00` | `--timestamp-format '%d.%m.%Y %H:%M:%S' --tz Europe/Warsaw` |
| przecinek dziesiętny i średnik, np. `21,5` | `--decimal ,` |
| inna nazwa kolumny czasu | `--timestamp-col Czas` |

Czas jest zapisywany w UTC (znacznik ze strefą, np. `Z` lub `+01:00`, ma pierwszeństwo przed `--tz`).
Puste lub niepoprawne komórki są odrzucane i policzone w raporcie. Nazwy czujników są
normalizowane (`Temperatura łożyska [°C]` → `temperatura_lozyska_c`) i dopisywane do `profile.yaml`.

## Excel: jeden plik na start (półautomatyczne wejście)

```bash
uv run diagsys xlsx-template szablon.xlsx            # przykładowy plik ze wszystkimi arkuszami
uv run diagsys import-xlsx pompa-01 szablon.xlsx --tz Europe/Warsaw
```

| Arkusz | Wymagany | Zawartość |
| --- | --- | --- |
| `dane` | tak | pierwsza kolumna = czas, każda następna = jeden czujnik (liczby) |
| `maszyna` | nie | dwie kolumny (parametr, wartość): nazwa, typ, opis, notatki |
| `czujniki` | nie | id, nazwa, jednostka, min, max (zakres używany przez kontrolę jakości) |
| `tryby_awarii` | nie | id, nazwa, opis, koszt_przestoju_na_godzine (dla preskrypcji) |
| `historia` | nie | czas, tryb_awarii, opis (znane awarie) |
| `kontekst` | nie | wolny tekst, jeden akapit na komórkę kolumny A, zapis do `docs/kontekst.md` |

Nazwy arkuszy mogą być po polsku lub angielsku. Maszyna jest tworzona, jeśli jeszcze nie istnieje.
Import używa tych samych reguł co CSV (powtórzenie niczego nie dubluje, strefy czasowe, odrzucone
komórki są liczone). Błędny arkusz nie zmienia niczego w bazie ani w profilu.

## Eksperymenty (powtarzalność)

`diagsys.eval.start_run` tworzy folder `experiments/<run_id>/` maszyny z `manifest.json`: ziarno,
konfiguracja, odcisk danych, wersje bibliotek i commit. `Run.finish(metrics)` zapisuje metryki.
Schemat architektury: `docs/architektura.md`.

## Jakość, czyszczenie, cechy (K3)

```bash
uv run diagsys quality press-01     # raport jakości, nic nie zapisuje do plików
uv run diagsys clean press-01       # + processed/clean.parquet, quality_report.json
uv run diagsys features press-01    # + processed/features.parquet (okno 1 h, krok 10 min)
```

Wszystko liczy się od zera z bazy (deterministycznie), więc trzy pliki z jednego uruchomienia
opisują ten sam stan danych. Surowe dane w bazie nie są zmieniane.

| Co wykrywa | Definicja |
| --- | --- |
| braki | brak odczytu czujnika w węźle siatki czasu (krok = typowy odstęp próbek) |
| skoki | pojedyncze odczyty daleko od mediany sąsiadów (filtr Hampela, 6 odchyleń robust); trwała zmiana poziomu to nie skok |
| zawieszone czujniki | co najmniej 20 identycznych odczytów pod rząd (czujniki dyskretne i stałe są pomijane i oznaczane osobno) |
| luki w czasie | przerwa dłuższa niż 5 kroków bez danych z żadnego czujnika; dzieli dane na segmenty |
| poza zakresem | odczyty poza `min_value`/`max_value` z `profile.yaml` (jeśli podane) |

Czyszczenie: skoki, zawieszone i poza zakresem odczyty stają się NaN; dziury do 5 punktów są
uzupełniane interpolacją (flaga `imputed`), dłuższych nie wypełniamy. Flagi każdego odczytu są
w `clean.parquet` (kolumny `<czujnik>__flags`, bity: 1 brak, 2 skok, 4 zawieszony, 8 poza zakresem,
16 uzupełniony).

Cechy (`features.parquet`): dla każdego okna i czujnika `mean, std, min, max, rms, slope_per_h,
kurtosis` (kolumny `<czujnik>__<cecha>`), plus `window_start`, `window_end`, `segment_id`,
`n_points`, `valid_frac`. Okno nigdy nie przechodzi przez lukę i jest podpisane czasem ostatniego
odczytu (bez patrzenia w przyszłość). Okno z mniej niż 80% ważnych odczytów czujnika ma dla niego NaN.

## Diagnoza (K4)

```bash
uv run diagsys diagnose press-01 --method iforest --exclude-sensor load_pct
uv run diagsys experiment-diagnosis press-01 --exclude-sensor load_pct   # porównanie z awariami
```

`diagnose` liczy cechy okienkowe z zapisanych odczytów, ocenia okna dwiema metodami
(`robust_z`: reguła progowa jako punkt odniesienia, `iforest`: Isolation Forest) i podaje epizody
alarmowe wraz z czujnikami, które je wywołały. Wynik zapisuje jako `processed/diagnosis_<metoda>.json`
(kontrakt `AnomalyReport`). Założenie: maszyna jest zdrowa na początku każdego ciągłego segmentu
danych, pierwsze 25% okien to wzorzec odniesienia (opcja `--reference-frac`).

`experiment-diagnosis` porównuje metody z wczytanymi awariami (wyprzedzenie alarmu, fałszywe alarmy)
i zapisuje powtarzalne uruchomienie w `experiments/`. Wielokrotne ziarna symulatora:
`uv run python scripts/diagnosis_study.py --seeds 101-110`. Pierwsze wyniki: `docs/wyniki_k4.md`.

## Predykcja RUL (K5)

```bash
uv run diagsys predict press-01 --exclude-sensor load_pct               # prognoza "teraz"
uv run diagsys predict press-01 --at 2026-03-01T12:00 --method age_baseline   # uczciwy backtest
uv run diagsys experiment-prediction press-01 --exclude-sensor load_pct   # ocena leave-one-cycle-out
```

`predict` uczy model na zapisanych awariach (arkusz `historia` albo `ingest-events`) i podaje medianę
pozostałego czasu życia, przedział 80% i przybliżone prawdopodobieństwo awarii w horyzoncie.
Z `--at` model widzi tylko dane i awarie do wskazanej chwili. Ostrzeżenia mówią wprost, gdy awarii
w historii jest za mało (poniżej 3 cykli prognoza jest niewiarygodna). Wynik trafia do
`processed/forecast_<metoda>.json` (kontrakt `Forecast`). Metody: `age_baseline` (bez czujników) i
`quantile_gbm`. Pierwsze wyniki, także niekorzystne: `docs/wyniki_k5.md`. Badanie na wielu ziarnach:
`uv run python scripts/prediction_study.py --seeds 101-110`.

## Dokumentacja maszyny i modele (K6)

```bash
# pliki .txt, .md, .pdf, .docx wrzucasz do data/machines/<maszyna>/docs/ (kontekst z Excela też tam trafia)
uv run diagsys docs index press-01                    # przyrostowo, tylko zmienione pliki
uv run diagsys docs search press-01 "wymiana łożyska" -k 5
uv run diagsys llm check                              # czy serwer modelu odpowiada
uv run diagsys docs ask press-01 "jak często wymieniać łożyska?"   # wymaga modelu
```

Indeks każdej maszyny jest osobny (`models/rag.duckdb`, wyszukiwanie dokładne po podobieństwie
kosinusowym). Wyszukiwanie działa bez serwera. `docs ask` odpowiada wyłącznie z odnalezionych
fragmentów, wymaga cytowania ich numerów, a kod sprawdza, czy numery istnieją: odpowiedź bez
poprawnego źródła jest oznaczona ostrzeżeniem, a przy braku fragmentów model nie jest w ogóle
wywoływany. Uruchomienie Bielika (Ollama, llama.cpp), wybór embeddingów i ograniczenia:
`docs/modele.md`.

## Struktura

```
src/diagsys/
  contracts/     typowane kontrakty (interfejsy I1-I7)
  ports.py       Storage, VectorIndex, LLMClient (wymienialne implementacje)
  gateway/       symulator danych (później MQTT/OPC UA, tylko odczyt)
  data/          import plików (ingest), jakość, czyszczenie, cechy (frame, quality, clean, features)
  storage/       DuckDB (implementacja portu Storage)
  config.py      ustawienia (zmienne DIAGSYS_*, plik .env)
  machines.py    profile maszyn
  doctor.py      sprawdzenie środowiska
  cli.py         polecenie `diagsys`
  tools/ rag/ llm/ agents/ orchestrator/ eval/   (kolejne kroki)
tests/
```

## Stan prac

| Krok | Zakres | Stan |
| --- | --- | --- |
| K1 | Szkielet, kontrakty, porty, profile maszyn, `doctor` | zrobiony |
| K2 | Dane syntetyczne, import plików, baza DuckDB | zrobiony |
| K3 | Walidacja, czyszczenie danych i cechy | zrobiony |
| A | Import Excela, szkielet powtarzalnych eksperymentów, schemat architektury | zrobiony |
| K4 | Diagnoza: anomalie, epizody alarmów, czujniki-sprawcy | zrobiony (hipotezy przyczyn: później) |
| K5 | Predykcja (RUL) z przedziałem, punkt odniesienia, ocena | zrobiony (kalibracja przedziałów: później) |
| K6 | Serwer modeli (Bielik) i RAG per maszyna | zrobiony (z modelem testowany tylko na atrapie serwera) |
| K7 | Preskrypcja | |
| K8 | Orkiestrator, raport, kontrola, interfejs | |
| K9-K11 | Ewaluacja, utwardzenie, pilot | |
