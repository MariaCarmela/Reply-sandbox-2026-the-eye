# Reply Sandbox 2026 — The Eye

> **Sistema Multi-Agente AI per la classificazione del rischio cittadino nella città digitale di Reply Mirror**

---

## Indice

1. [Panoramica del Progetto](#1-panoramica-del-progetto)
2. [Contesto della Challenge](#2-contesto-della-challenge)
3. [Architettura Multi-Agente](#3-architettura-multi-agente)
4. [Struttura del Progetto](#4-struttura-del-progetto)
5. [Descrizione degli Agenti](#5-descrizione-degli-agenti)
6. [Notebook di Analisi e Modellazione](#6-notebook-di-analisi-e-modellazione)
7. [Stack Tecnologico](#7-stack-tecnologico)
8. [Setup e Installazione](#8-setup-e-installazione)
9. [Pipeline di Esecuzione](#9-pipeline-di-esecuzione)
10. [Generazione della Submission](#10-generazione-della-submission)
11. [Tracking con Langfuse](#11-tracking-con-langfuse)
12. [Risultati per Livello](#12-risultati-per-livello)
13. [Configurazione Avanzata](#13-configurazione-avanzata)
14. [Note e Limitazioni](#14-note-e-limitazioni)

---

## 1. Panoramica del Progetto

**The Eye** è un sistema multi-agente basato su intelligenza artificiale sviluppato per la challenge **Reply Sandbox 2026**. Il sistema opera nella città digitale di **Reply Mirror**, ambientata nell'anno 2087, con l'obiettivo di analizzare il comportamento dei cittadini digitali e classificarli come **a rischio** (`label=1`) o **standard** (`label=0`), al fine di attivare percorsi di supporto preventivo.

### Obiettivo Principale

```
Classificazione binaria per cittadino:
  - label = 1  →  Cittadino a rischio  →  Percorso di supporto preventivo
  - label = 0  →  Cittadino standard   →  Nessuna azione richiesta
```

### Punti Chiave

- **Approccio**: Pipeline orchestrata da agenti specializzati, ognuno responsabile di una fase del processo
- **Metrica principale**: F1 Score (bilanciamento tra precision e recall)
- **Sfida**: Dataset fortemente sbilanciato (pochi positivi rispetto ai negativi)
- **Tracking**: Ogni sessione di esecuzione viene tracciata tramite **Langfuse** per garantire riproducibilità e audit

---

## 2. Contesto della Challenge

### Reply Mirror — Anno 2087

Nel 2087, **Reply Mirror** è una città completamente digitale in cui ogni cittadino genera una traccia continua di eventi comportamentali, spostamenti geografici e dati di salute. Il sistema **The Eye** analizza questi flussi di dati per identificare precocemente i cittadini che potrebbero beneficiare di interventi preventivi.

### Livelli della Challenge

| Livello | Cittadini | Eventi | Positivi Noti | Stato |
|---------|-----------|--------|----------------|-------|
| `lev_1` | 5 | 50 | 1 (WNACROYX) | ✅ Completato |
| `lev_2` | TBD | TBD | TBD | 🔄 In corso |
| `lev_3` | TBD | TBD | TBD | 🔜 Pianificato |

### Caso Noto — Livello 1

> Il cittadino **Craig Connor** (`CitizenID: WNACROYX`) è l'unico classificato come a rischio nel livello 1, come da specifica nel file `personas.md`.

---

## 3. Architettura Multi-Agente

### Schema Architetturale

```
╔══════════════════════════════════════════════════════════════════════╗
║                        THE EYE — ORCHESTRATORE                       ║
║                    (src/agents/the_eye.py)                            ║
║                                                                        ║
║   ┌─────────────┐    ┌──────────────┐    ┌─────────────────────────┐  ║
║   │  Langfuse   │◄───│  Session ID  │───►│   Trace & Span Logger   │  ║
║   │  Tracking   │    │  (per submit)│    │   (src/tracking.py)     │  ║
║   └─────────────┘    └──────────────┘    └─────────────────────────┘  ║
╚══════════════════════════════════════════════════════════════════════╝
                                  │
                    ┌─────────────▼─────────────┐
                    │         DataAgent          │
                    │  ─────────────────────     │
                    │  • Carica status.csv        │
                    │  • Carica locations.json    │
                    │  • Carica users.json        │
                    │  • Assegna label manuali    │
                    │    da personas.md           │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │                           │
          ┌─────────┴──────────┐   ┌────────────┴──────────┐
          │    FeatureAgent    │   │       GeoAgent         │
          │  ──────────────    │   │  ───────────────────   │
          │  • Feature temp.   │   │  • Centroidi città     │
          │  • Rolling avg     │   │  • Distanza Haversine  │
          │  • Lag features    │   │  • Aggregati geo per   │
          │  • One-hot encode  │   │    cittadino           │
          │  • Health agg.     │   └────────────┬──────────┘
          └─────────┬──────────┘                │
                    │                           │
                    └─────────────┬─────────────┘
                                  │ (feature merge)
                    ┌─────────────▼─────────────┐
                    │      PredictionAgent       │
                    │  ─────────────────────     │
                    │  • XGBoost Classifier      │
                    │  • Leave-One-Citizen-Out   │
                    │    Cross Validation        │
                    │  • scale_pos_weight        │
                    │    (class imbalance)       │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │       OutputAgent          │
                    │  ─────────────────────     │
                    │  • Genera file .txt UTF-8  │
                    │  • Lista CitizenID a       │
                    │    rischio (label=1)        │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │      submission_lev*.txt   │
                    │      output/               │
                    └───────────────────────────┘
```

### Flusso Dati

```
data/raw/
  └── public_lev_*/
        ├── status.csv        ──┐
        ├── locations.json    ──┤──► DataAgent ──► DataFrame unificato
        ├── users.json        ──┤
        └── personas.md       ──┘
                                          │
                              ┌───────────┴───────────┐
                              ▼                       ▼
                        FeatureAgent             GeoAgent
                              │                       │
                              └───────────┬───────────┘
                                          ▼
                                  Feature Matrix
                                  (features_lev*.csv)
                                          │
                                          ▼
                                  PredictionAgent
                                  (xgb_lev*.pkl)
                                          │
                                          ▼
                                   OutputAgent
                                          │
                                          ▼
                              output/submission_lev*.txt
```

---

## 4. Struttura del Progetto

```
reply-sandbox-2026-the-eye/
│
├── 📁 data/
│   ├── 📁 raw/
│   │   ├── 📁 public_lev_1/
│   │   │   ├── status.csv          # Dati di stato/eventi dei cittadini
│   │   │   ├── locations.json      # Coordinate geografiche degli eventi
│   │   │   ├── users.json          # Anagrafica e profilo dei cittadini
│   │   │   └── personas.md         # Descrizioni narrative + label manuali
│   │   ├── 📁 public_lev_2/
│   │   │   ├── status.csv
│   │   │   ├── locations.json
│   │   │   ├── users.json
│   │   │   └── personas.md
│   │   └── 📁 public_lev_3/
│   │       ├── status.csv
│   │       ├── locations.json
│   │       ├── users.json
│   │       └── personas.md
│   │
│   └── 📁 processed/
│       └── features_lev1.csv       # Feature matrix elaborata (lev_1)
│
├── 📁 models/
│   └── xgb_lev1.pkl                # Modello XGBoost serializzato (lev_1)
│
├── 📁 notebooks/
│   ├── 01_eda.py                   # Exploratory Data Analysis
│   ├── 02_feature_engineering.py   # Costruzione delle feature
│   ├── 03_model_training.py        # Training e validazione del modello
│   └── 04_submission.py            # Generazione file di submission
│
├── 📁 src/
│   ├── 📁 agents/
│   │   └── the_eye.py              # Orchestratore + tutti gli agenti
│   └── main.py                 # Modulo Langfuse tracking
│
├── 📁 output/
│   ├── submission_lev1.txt         # Submission livello 1
│   ├── submission_lev2.txt         # Submission livello 2
│   └── submission_lev3.txt         # Submission livello 3
│
├── 📁 docs/
│   └── DOCUMENTATION.md            # Questo file
│
├── requirements.txt                # Dipendenze Python
├── .gitignore                      # File/cartelle esclusi da git
└── README.md                       # Guida rapida al progetto
```

### Descrizione dei File di Dati

| File | Formato | Contenuto |
|------|---------|-----------|
| `status.csv` | CSV | Registro eventi per cittadino: timestamp, tipo evento, metriche di salute |
| `locations.json` | JSON | Coordinate GPS degli eventi: latitudine, longitudine, nome luogo |
| `users.json` | JSON | Profilo cittadino: età, professione, caratteristiche demografiche |
| `personas.md` | Markdown | Descrizioni narrative dei cittadini con indicazioni sul rischio |

---

## 5. Descrizione degli Agenti

### 5.1 DataAgent

**File**: `src/agents/the_eye.py` → classe `DataAgent`

**Responsabilità**: Caricamento, validazione e unificazione dei dati grezzi.

#### Operazioni Principali

```python
# Pseudocodice delle operazioni del DataAgent
class DataAgent:
    def load_status(self, path: str) -> pd.DataFrame
    def load_locations(self, path: str) -> pd.DataFrame
    def load_users(self, path: str) -> pd.DataFrame
    def parse_personas(self, path: str) -> dict[str, int]  # {citizen_id: label}
    def merge_data(self) -> pd.DataFrame
    def assign_labels(self) -> pd.DataFrame
```

#### Input / Output

| Input | Tipo | Descrizione |
|-------|------|-------------|
| `status.csv` | CSV | Dati evento per riga |
| `locations.json` | JSON | Coordinate geografiche |
| `users.json` | JSON | Profilo utente |
| `personas.md` | Markdown | Label manuali da narrative |

| Output | Tipo | Descrizione |
|--------|------|-------------|
| `df_merged` | `pd.DataFrame` | Dataset unificato con colonna `label` |

#### Logica di Parsing di `personas.md`

Il DataAgent estrae le label di rischio dal file `personas.md` attraverso pattern matching sul testo narrativo. Le keyword utilizzate per l'identificazione del rischio includono termini come *"at risk"*, *"anomalous behavior"*, *"critical"* e i mapping espliciti per i cittadini noti.

```python
# Esempio di mapping estratto da personas.md (lev_1)
MANUAL_LABELS = {
    "WNACROYX": 1,  # Craig Connor — a rischio
    # tutti gli altri → 0 (default)
}
```

---

### 5.2 FeatureAgent

**File**: `src/agents/the_eye.py` → classe `FeatureAgent`

**Responsabilità**: Costruzione della feature matrix a partire dal dataset unificato.

#### Feature Implementate

**Feature Temporali**
```
- hour_of_day          : ora dell'evento (0–23)
- day_of_week          : giorno della settimana (0=Lunedì, 6=Domenica)
- is_weekend           : flag binario (0/1)
- hour_sin / hour_cos  : encoding ciclico dell'ora (sin/cos)
```

**Rolling Averages** (finestre: 3, 7, 14 eventi)
```
- rolling_mean_health_{w}   : media mobile metriche di salute
- rolling_std_health_{w}    : deviazione standard mobile
- rolling_count_events_{w}  : numero eventi nella finestra
```

**Lag Features** (lag: 1, 2, 3 eventi precedenti)
```
- lag_{n}_health_metric     : valore metrica di salute n eventi fa
- lag_{n}_event_type        : tipo evento n passi fa
```

**One-Hot Encoding — EventType**
```
- event_type_SOCIAL         : evento di tipo sociale
- event_type_HEALTH         : evento di tipo salute
- event_type_MOVEMENT       : evento di tipo spostamento
- event_type_*              : altri tipi presenti nel dataset
```

**Health Aggregates** (per cittadino)
```
- mean_health_score         : media complessiva del punteggio salute
- std_health_score          : variabilità del punteggio salute
- min_health_score          : valore minimo registrato
- max_health_score          : valore massimo registrato
- trend_health_score        : pendenza lineare (regressione OLS)
```

---

### 5.3 GeoAgent

**File**: `src/agents/the_eye.py` → classe `GeoAgent`

**Responsabilità**: Analisi geospaziale degli spostamenti e dei pattern di localizzazione.

#### Algoritmi e Calcoli

**Centroidi per Cittadino**
```python
# Calcolo del centroide come media pesata delle coordinate visitate
centroid_lat = df.groupby("citizen_id")["latitude"].mean()
centroid_lon = df.groupby("citizen_id")["longitude"].mean()
```

**Distanza Haversine**
```python
# Formula di Haversine per la distanza tra due punti GPS
def haversine(lat1, lon1, lat2, lon2) -> float:
    """
    Restituisce la distanza in chilometri tra due coordinate.
    Utilizzata per calcolare:
    - distanza media dal centroide
    - distanza massima percorsa in un evento
    - raggio di movimento del cittadino
    """
```

#### Feature Geospaziali Prodotte

| Feature | Descrizione |
|---------|-------------|
| `dist_from_centroid` | Distanza media di ogni evento dal centroide del cittadino |
| `max_dist_event` | Distanza massima registrata in un singolo evento |
| `movement_radius` | Raggio complessivo di mobilità (km) |
| `n_unique_locations` | Numero di luoghi unici visitati |
| `geo_entropy` | Entropia della distribuzione delle visite per luogo |
| `home_cluster_dist` | Distanza media dal cluster principale (casa/lavoro) |

---

### 5.4 PredictionAgent

**File**: `src/agents/the_eye.py` → classe `PredictionAgent`

**Responsabilità**: Addestramento del modello di classificazione e generazione delle predizioni.

#### Modello

```python
XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=<ratio_neg/ratio_pos>,  # gestione sbilanciamento
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42
)
```

#### Strategia di Validazione — Leave-One-Citizen-Out CV

Data la natura del problema (pochi cittadini, dataset piccolo), viene utilizzata una strategia **Leave-One-Citizen-Out Cross Validation**:

```
Iterazione 1: train=[C2, C3, C4, C5],  test=[C1]
Iterazione 2: train=[C1, C3, C4, C5],  test=[C2]
Iterazione 3: train=[C1, C2, C4, C5],  test=[C3]
Iterazione 4: train=[C1, C2, C3, C5],  test=[C4]
Iterazione 5: train=[C1, C2, C3, C4],  test=[C5]
```

> Questa strategia evita il **data leakage** che si verificherebbe con una split casuale sulle righe, poiché gli eventi dello stesso cittadino sono correlati nel tempo.

#### Gestione dello Sbilanciamento

```python
# Calcolo automatico del peso per la classe positiva
n_neg = sum(y == 0)
n_pos = sum(y == 1)
scale_pos_weight = n_neg / n_pos  # Es: 4/1 = 4.0 per lev_1
```

#### Metriche di Valutazione

```python
from sklearn.metrics import f1_score, precision_score, recall_score

metrics = {
    "f1":        f1_score(y_true, y_pred),           # metrica principale
    "precision": precision_score(y_true, y_pred),
    "recall":    recall_score(y_true, y_pred),
}
```

---

### 5.5 OutputAgent

**File**: `src/agents/the_eye.py` → classe `OutputAgent`

**Responsabilità**: Serializzazione dei risultati nel formato richiesto dalla piattaforma.

#### Formato Output

Il file di submission è un file di testo **UTF-8** contenente un `CitizenID` per riga, corrispondente ai soli cittadini classificati come `label=1`:

```
# Esempio: output/submission_lev1.txt
WNACROYX
```

#### Logica di Generazione

```python
def generate_submission(predictions: dict, output_path: str, level: int):
    """
    Genera il file di submission per il livello specificato.
    
    Args:
        predictions: {citizen_id: label} per tutti i cittadini
        output_path: percorso del file di output
        level: numero del livello (1, 2, 3)
    
    Output:
        File UTF-8 con un CitizenID per riga (solo label=1)
    """
    at_risk = [cid for cid, label in predictions.items() if label == 1]
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(at_risk))
```

---

### 5.6 TheEye — Orchestratore

**File**: `src/agents/the_eye.py` → classe `TheEye`

**Responsabilità**: Coordinamento dell'intera pipeline, gestione degli errori e integrazione con Langfuse.

#### Flusso di Orchestrazione

```python
class TheEye:
    def run(self, level: int):
        # 1. Inizializzazione sessione Langfuse
        session = self.tracker.start_session(level)
        
        # 2. Caricamento dati
        with self.tracker.span("data_loading"):
            data = self.data_agent.load(level)
        
        # 3. Feature engineering (parallelo)
        with self.tracker.span("feature_engineering"):
            features = self.feature_agent.transform(data)
            geo_features = self.geo_agent.transform(data)
            X = pd.concat([features, geo_features], axis=1)
        
        # 4. Training e predizione
        with self.tracker.span("prediction"):
            predictions = self.prediction_agent.fit_predict(X, data.labels)
        
        # 5. Generazione output
        with self.tracker.span("output_generation"):
            self.output_agent.generate(predictions, level)
        
        # 6. Log metriche finali su Langfuse
        self.tracker.log_metrics(predictions, data.labels)
        self.tracker.end_session(session)
```

---

## 6. Notebook di Analisi e Modellazione

### 6.1 `01_eda.py` — Exploratory Data Analysis

**Scopo**: Comprensione approfondita dei dati grezzi prima di qualsiasi trasformazione.

**Contenuto**:
- Statistiche descrittive su `status.csv` (distribuzioni, missing values, outlier)
- Visualizzazione della distribuzione temporale degli eventi per cittadino
- Analisi delle distribuzioni geografiche tramite scatter plot GPS
- Heatmap delle correlazioni tra variabili numeriche
- Profili narrativi estratti da `personas.md`
- Analisi dello sbilanciamento delle classi
- Visualizzazioni interattive con **Plotly** e statiche con **Seaborn**

**Output principali**:
```
- Distribuzione eventi per cittadino
- Timeline degli eventi per tipo (EventType)
- Mappa 2D degli spostamenti per cittadino
- Matrice di correlazione delle metriche di salute
```

---

### 6.2 `02_feature_engineering.py` — Feature Engineering

**Scopo**: Costruzione e validazione della feature matrix.

**Contenuto**:
- Instanziazione e test di `FeatureAgent` e `GeoAgent`
- Analisi dell'importanza delle feature (permutation importance)
- Verifica dell'assenza di data leakage nelle feature temporali
- Export del dataset processato in `data/processed/features_lev1.csv`
- Analisi della distribuzione delle feature per classe (0 vs 1)

**Output principali**:
```
data/processed/features_lev1.csv    # Feature matrix completa
```

---

### 6.3 `03_model_training.py` — Training del Modello

**Scopo**: Addestramento, validazione e selezione del modello finale.

**Contenuto**:
- Comparazione tra XGBoost, LightGBM e altri classificatori
- Leave-One-Citizen-Out CV con reportistica per ogni fold
- Ottimizzazione degli iperparametri (Grid Search / Bayesian)
- Feature importance plots
- Analisi delle curve Precision-Recall
- Serializzazione del modello finale in `models/xgb_lev1.pkl`
- Logging su Langfuse delle metriche di ogni esperimento

**Output principali**:
```
models/xgb_lev1.pkl     # Modello serializzato
- F1 Score per fold
- Precision-Recall Curve
- Feature Importance Bar Chart
```

---

### 6.4 `04_submission.py` — Generazione Submission

**Scopo**: Generazione del file finale di submission a partire dal modello addestrato.

**Contenuto**:
- Caricamento del modello da `models/xgb_lev1.pkl`
- Predizione su tutto il dataset
- Validazione del formato di output (UTF-8, un ID per riga)
- Scrittura di `output/submission_lev1.txt`
- Stampa del `Session ID` Langfuse da inserire nella piattaforma

**Output principali**:
```
output/submission_lev1.txt    # File di submission finale
```

---

## 7. Stack Tecnologico

### Dipendenze Principali

| Libreria | Versione | Utilizzo |
|----------|----------|---------|
| `python` | 3.13+ | Runtime principale |
| `xgboost` | latest | Modello di classificazione principale |
| `lightgbm` | latest | Modello alternativo per comparazione |
| `scikit-learn` | latest | Preprocessing, CV, metriche |
| `pandas` | latest | Manipolazione dati tabulari |
| `numpy` | latest | Operazioni numeriche |
| `scipy` | latest | Calcoli statistici (regressione OLS per trend) |
| `geopandas` | latest | Analisi geospaziale avanzata |
| `langchain` | latest | Framework per orchestrazione agenti |
| `langfuse` | latest | Tracking e observability della pipeline |
| `plotly` | latest | Visualizzazioni interattive |
| `seaborn` | latest | Visualizzazioni statistiche |
| `matplotlib` | latest | Plot di base |

### Dipendenze di Sistema

```
Python 3.13+
pip o conda
Git
```

---

## 8. Setup e Installazione

### 8.1 Prerequisiti

- Python **3.13** o superiore
- `pip` aggiornato all'ultima versione
- Accesso alla piattaforma **Reply Sandbox 2026**
- Credenziali **Langfuse** (fornite dalla challenge)

### 8.2 Clonare il Repository

```bash
git clone https://github.com/<your-org>/reply-sandbox-2026-the-eye.git
cd reply-sandbox-2026-the-eye
```

### 8.3 Creare l'Ambiente Virtuale

```bash
# Con venv (raccomandato)
python3.13 -m venv .venv
source .venv/bin/activate          # Linux/macOS
.venv\Scripts\activate             # Windows

# Con conda
conda create -n the-eye python=3.13
conda activate the-eye
```

### 8.4 Installare le Dipendenze

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 8.5 Configurare le Variabili d'Ambiente

Creare un file `.env` nella root del progetto:

```bash
# .env
LANGFUSE_PUBLIC_KEY=pk-lf-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
LANGFUSE_SECRET_KEY=sk-lf-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
LANGFUSE_HOST=https://challenges.reply.com/langfuse
```

> ⚠️ Non committare mai il file `.env` su Git. È già incluso nel `.gitignore`.

### 8.6 Verificare l'Installazione

```bash
python -c "import xgboost, langfuse, geopandas; print('Setup OK ✓')"
```

### 8.7 Contenuto di `requirements.txt`

```txt
xgboost>=2.0.0
lightgbm>=4.0.0
scikit-learn>=1.4.0
pandas>=2.2.0
numpy>=1.26.0
scipy>=1.12.0
geopandas>=0.14.0
langchain>=0.1.0
langfuse>=2.0.0
plotly>=5.18.0
seaborn>=0.13.0
matplotlib>=3.8.0
python-dotenv>=1.0.0
joblib>=1.3.0
tqdm>=4.66.0
```

---

## 9. Pipeline di Esecuzione

### 9.1 Esecuzione Completa (Raccomandata)

Per eseguire l'intera pipeline orchestrata da **TheEye**:

```bash
python src/agents/the_eye.py --level 1
```

**Parametri disponibili**:

```bash
python src/agents/the_eye.py \
    --level 1 \                    # livello da processare (1, 2, 3)
    --data-dir data/raw \          # directory dati grezzi
    --output-dir output \          # directory output
    --save-model \                 # salva il modello in models/
    --verbose                      # abilita output dettagliato
```

### 9.2 Esecuzione Step-by-Step tramite Notebook

```bash
# Step 1: EDA
python notebooks/01_eda.py

# Step 2: Feature Engineering
python notebooks/02_feature_engineering.py

# Step 3: Training
python notebooks/03_model_training.py

# Step 4: Submission
python notebooks/04_submission.py
```

### 9.3 Output Atteso durante l'Esecuzione

```
[TheEye] 🚀 Starting pipeline for Level 1
[TheEye] 📊 Session ID: sess_abc123xyz (save this!)
[DataAgent] ✓ Loaded 50 events for 5 citizens
[DataAgent] ✓ Labels assigned: {WNACROYX: 1, others: 0}