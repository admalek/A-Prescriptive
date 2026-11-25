# Modele lokalne: uruchomienie i sprawdzenie

Wszystko, co dotyczy modeli, jest opcjonalne: wyszukiwanie w dokumentacji maszyny działa bez
żadnego serwera (lokalne wektory z n-gramów znaków). Serwer modelu jest potrzebny do odpowiedzi na
pytania (`diagsys docs ask`) i, od K8, do opisów w raporcie.

**Stan weryfikacji:** klient i serwer-atrapa są przetestowane automatycznie. Z prawdziwym Bielikiem
i modelem embeddingów nie przetestowano nic, bo środowisko, w którym powstał kod, nie miało dostępu
do hostów z modelami. Pierwszy prawdziwy test to `diagsys llm check` u Ciebie.

## Ollama (najprościej)

```bash
ollama pull SpeakLeash/bielik-4.5b-v3.0-instruct       # tag z biblioteki Ollamy, sprawdź aktualny
ollama cp SpeakLeash/bielik-4.5b-v3.0-instruct bielik-4.5b-instruct   # nazwa, której oczekuje diagsys
uv run diagsys llm check
```

Zamiast aliasu możesz ustawić `DIAGSYS_LLM_MODEL=SpeakLeash/bielik-4.5b-v3.0-instruct`. Warianty
kwantyzacji (np. `:Q8_0`) są na stronie modelu w bibliotece Ollamy. Na laptopie bez GPU czas
odpowiedzi 4.5B bywa rzędu dziesiątek sekund, więc w razie potrzeby podnieś `DIAGSYS_LLM_TIMEOUT_S`.

## llama.cpp

```bash
llama-server -m bielik-4.5b-v3.0-instruct.Q4_K_M.gguf --port 8080
export DIAGSYS_LLM_BASE_URL=http://localhost:8080/v1
```

(plik GGUF pobierasz z Hugging Face; nazwa pliku zależy od wydania).

## Embeddingi

| `DIAGSYS_EMBEDDING_BACKEND` | Co robi | Zalety i wady |
| --- | --- | --- |
| `hash` (domyślnie) | lokalne wektory z n-gramów znaków, bez modelu | działa offline, znosi odmianę i brak polskich znaków, ale dopasowuje brzmienie wyrazów, nie znaczenie |
| `server` | wektory z serwera modelu (`DIAGSYS_EMBEDDING_MODEL`) | rozumie znaczenie, wymaga serwowanego modelu embeddingów z obsługą polskiego |

Dla modeli `e5` kod sam dodaje przedrostki `query:` i `passage:`, których ten model wymaga. Nazwy
modelu embeddingów dostępnego w Twojej Ollamie nie sprawdzałem: ustaw `DIAGSYS_EMBEDDING_MODEL` na
taki, który serwer faktycznie wystawia (`diagsys llm check` wypisze listę).

Indeks zapamiętuje, czym został zbudowany. Po zmianie backendu lub modelu trzeba go zbudować od
nowa: `diagsys docs index <maszyna> --rebuild`. Bez tego komenda odmówi z czytelnym komunikatem.

## Jakość wyszukiwania (lokalne wektory)

Na małym zbiorze testowym (4 dokumenty o pompie, 14 pytań z innymi formami wyrazów niż w tekście)
lokalne wektory trafiały we właściwy dokument jako pierwszy wynik w 13 z 14 pytań. Jedyna pomyłka to
pytanie bez wspólnych rdzeni słów z dokumentem („pompa hałasuje jak żwir" wobec opisu kawitacji).
To mały, napisany na potrzeby testu zbiór, więc wynik jest orientacyjny. Niskie wyniki podobieństwa
(poniżej ok. 0,15) oznaczają słabe dopasowanie. Dla prawdziwej dokumentacji potrzebna jest ocena
na własnych pytaniach (K9).
