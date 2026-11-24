# Architektura systemu diagsys

Legenda: zielony = gotowe (K1–K3, import Excela, szkielet eksperymentów), żółty = następny krok (K4),
szary = zaplanowane, czerwony = decyzja człowieka.

Zasady: model językowy planuje i wyjaśnia, wszystkie liczby liczą narzędzia deterministyczne,
system tylko doradza (brak zapisu do maszyn), a każda teza w raporcie ma źródło.

Potok deterministyczny jest jednocześnie punktem odniesienia w eksperymentach (H1–H4): warstwa
agentowa działa na tych samych narzędziach, więc porównanie "potok bez agentów a system
wieloagentowy" mierzy wkład agentów, a nie różnicę w obliczeniach.

```mermaid
flowchart TB
    classDef done fill:#d8f0dc,stroke:#2f7d3b,color:#12301a
    classDef next fill:#fff1cc,stroke:#b58105,color:#3d2c00
    classDef later fill:#e8e8ee,stroke:#6b6b7a,color:#25252e
    classDef human fill:#fde0d6,stroke:#b5471f,color:#3d1507
    classDef grp fill:#ffffff,stroke:#9a9aa8,color:#25252e

    subgraph AGENTS["Warstwa wieloagentowa (K8): planuje, wywołuje narzędzia, pisze raport"]
        direction LR
        O["Orkiestrator"]:::later
        A1["Agent<br/>danych"]:::later
        A2["Agent<br/>diagnozy"]:::later
        A3["Agent<br/>predykcji"]:::later
        A4["Agent<br/>preskrypcji"]:::later
        A5["Agent<br/>raportu"]:::later
        O --> A1 & A2 & A3 & A4 & A5
    end
    class AGENTS grp

    LLM["Model językowy (I5)<br/>Bielik lokalnie: plan, wyjaśnienia"]:::later
    RAG[("RAG (K6)<br/>dokumentacja, kontekst")]:::later
    AGENTS <--> LLM
    AGENTS <--> RAG

    subgraph PIPE["Potok deterministyczny: wszystkie liczby liczą narzędzia, nie model"]
        direction TB
        X["Excel / CSV: dane, opis, historia, kontekst"]:::done
        GW["Import i walidacja (I1)"]:::done
        DB[("DuckDB + Parquet")]:::done
        PREP["Jakość → czyszczenie → cechy okienkowe (K3)"]:::done
        D["Diagnoza (K4): anomalie + reguły"]:::next
        P["Predykcja RUL (K5): trend + niepewność"]:::later
        R["Preskrypcja (K7): decyzja i koszt zwłoki"]:::later
        X --> GW --> DB --> PREP --> D --> P --> R
    end
    class PIPE grp

    AGENTS ==>|"wywołują narzędzia"| PIPE
    H{{"Nadzorca człowiek: zatwierdza lub odrzuca"}}:::human
    REP["Raport: liczby z narzędzi, źródła tez"]:::later
    PIPE --> H --> REP

    EVAL["Ewaluacja (H1–H4): manifest uruchomienia, punkt odniesienia bez agentów, ablacje bez LLM, RAG i zatwierdzania"]:::done
    REP -.-> EVAL
```

Obraz: `docs/architektura.png` (render z `docs/architektura.mmd`: `mmdc -i docs/architektura.mmd -o docs/architektura.png -s 2`).
