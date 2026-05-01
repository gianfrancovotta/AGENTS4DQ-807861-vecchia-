# AGENTS4DQ — Agents for Data Quality

**REPLY × LUISS Machine Learning Project | A.Y. 2025/26**

**Team members:**
- Maria Dichio (822231)
- Gianfranco Votta (807861) — *Captain*
- Armando Fornario (813811)

---

## Section 1 — Introduction

NoiPA is the digital platform of the *Ministero dell'Economia e delle Finanze* that manages salaries, timesheets, and tax/social security obligations for employees of the Italian Public Administration. The platform periodically receives datasets from heterogeneous sources (CSV, JSON, databases) containing demographic, economic, and administrative data. Currently, data validation is either manual or entirely absent, which poses serious risks in terms of data reliability and downstream decision-making.

This project builds **AGENTS4DQ**, a multi-agent system that automatically ingests a raw CSV dataset and:

1. **Validates and corrects** the schema (data types and naming conventions)
2. **Analyzes completeness** (null/missing values, placeholder detection, sparse columns and rows)
3. **Checks consistency** (duplicate detection and removal, format uniformity, cross-column logical rules)
4. **Detects anomalies** (univariate statistical outliers and rare categorical values)
5. **Generates a remediation report** with a Data Reliability Score (0-100) and a Human-in-the-Loop Action Plan

The system produces a structured quality report and presents it through a **Streamlit** web interface with a typewriter-style streaming output. All code is also available in a single self-cointaned Jupyter notebook (`main.ipynb`).

However, we suggest to run ('main.py') in the src folder by typing in the terminal "streamlit run src/main.py" in order to see the graphical interface directly.

---

## Section 2 — Methods

### 2.1 Architecture

AGENTS4DQ follows a **Supervisor architecture** where a central orchestrator (`DataOrchestrator`) receives the user's request and coordinates five specialized agents in sequence:

![AGENTS4DQ System Architecture](images/architecture.png)

### 2.2 Agent Details

| Agent | LLM Backend | Key Responsibilities |
|---|---|---|
| **DataOrchestrator** | Gemma 4 31B IT | Parses user intent; loads CSV/XLSX into a DataFrame |
| **SchemaValidator** | Gemini 3.1 Flash Lite Preview | Checks column data types against regex pattern fingerprints; validates naming conventions; **automatically corrects** both (type casting + column renaming) |
| **CompletenessAnalyst** | Gemma 4 31B IT | Detects placeholder strings (e.g. "N/A", "-", "null") via LLM and replaces them with NaN; computes per-column and per-row missing percentages; flags sparse columns (>50% missing) and sparse rows for removal |
| **ConsistencyValidator** | Gemini 3.1 Flash Lite Preview | Identifies key columns and detects duplicates (exact + key-column); removes exact duplicates automatically; uses regex pattern analysis to flag format inconsistencies; validates cross-column logical relationships on a sample |
| **AnomalyDetector** | Gemma 4 31B IT | LLM selects candidate numerical columns → 3σ outlier detection; LLM selects candidate categorical columns → flags values appearing in <1% of rows |
| **RemediatorAgent** | Gemma 4 31B IT | Aggregates all findings into a final Human-in-the-Loop Action Plan with specific imputation/correction strategies and a Data Reliability Score (0–100) |

### 2.3 Design Choices

- **LLM-driven column selection**: rather than hard-coding which columns to validate, the agents ask the LLM to infer which columns are relevant for each check (e.g., which columns are numerical candidates for outlier detection). This makes the system **dataset-agnostic**.
- **Automatic corrections**: the SchemaValidator doesn't just detect issues, it automatically casts data types and renames columns based on LLM-generated correction mappings. High-confidence fixes (type casting, naming standardisation, deduplication) are applied automatically, while domain-dependent decisions are left to the human operator via the Remediator's Action Plan.
- **Regex pattern fingerprinting** (`functions.py`): each cell value is converted into a pattern string (digits → `N`, letters → `W`, punctuation preserved) to build a frequency map per column. This enables format consistency detection without manually defining expected formats.
- **Hybrid approach**: statistical checks (3σ for outliers, frequency thresholds for categorical anomalies, exact duplicate detection) are combined with LLM reasoning (placeholder identification, cross-column logic, naming conventions).
- **SQLite caching** (`langchain` `SQLiteCache`): LLM responses are cached to avoid redundant API calls during development and demo.
- **Streamlit frontend**: results are streamed with a typewriter effect for a more readable, step-by-step presentation.

### 2.4 Environment Setup

```bash
# Clone the repository
git clone https://github.com/gianfrancovotta/AGENTS4DQ-807861.git
cd AGENTS4DQ-807861

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

A `.env` file is required at the project root with the following key:

```
GOOGLE_API_KEY=<given_api_key_here>

LANGSMITH_API_KEY=<given_api_key_here>

LANGSMITH_TRACING=true

LANGSMITH_PROJECT=lc-essentials

LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com
```

To launch the application:

```bash
cd src
streamlit run main.py
```

To run the notebook:

```bash
jupyter notebook main.ipynb
```

---

## Section 3 — Experimental Design

### Experiment 1 — End-to-end Quality Assessment on `spesa.csv`

- **Purpose**: validate that the full agent pipeline correctly identifies schema issues, missing data, inconsistencies, and outliers on a real NoiPA expenditure dataset (7,543 rows, 18 columns, including intentionally noisy/redundant columns such as `SPESA TOTALE`, `2cod_imposta`, `cod imposta ext`, `ente%code`).
- **Baseline**: manual inspection of the dataset in a spreadsheet editor.
- **Evaluation Metrics**: qualitative comparison whether each agent detects the expected issues (e.g., redundant columns, naming convention violations, type mismatches, format inconsistencies in the pattern report).

### Experiment 2 — End-to-end Quality Assessment on `attivazioniCessazioni.csv`

- **Purpose**: test generalization to a structurally different NoiPA dataset (20,102 rows, 19 columns - personnel activations/terminations, with columns like `mese`, `anno`, `provincia_sede`, `attivazioni`, `cessazioni`, `qualifica`, and intentionally noisy columns such as `3descrizione`, `regione%sede`, `att ivazioni`).
- **Baseline**: manual inspection.
- **Evaluation Metrics**: same qualitative assessment, the system should flag naming violations (`3descrizione`, `regione%sede`, `att ivazioni`), detect cross-column logic (e.g., `mese`/`anno` coherence with `RATA`), spot province casing inconsistencies (e.g., `Aq` vs `AQ`), and identify redundant mirror columns.

---

## Section 4 — Results

### Key Findings

**Schema Validation**: the SchemaValidator correctly identifies type mismatches (e.g., numeric columns stored as string/objects) and naming violations (columns with special characters like `%`, leading digits, or spaces such as `ente%code`, `3descrizione`, `att ivazioni`). Automatic corrections are successfully applied: types are cast and column names are standardised.

**Completeness Analysis**: the CompletenessAnalyst detects placeholder values across both datasets and computes per-column and per-row missing rates. Columns like `note` and `fonte_dato` are flagged as largely empty (>50% missing) and candidates for removal. Row-level analysis identifies individual records with excessive missing data.

**Consistency Validation**: the regex pattern fingerprinting approach proves effective at detecting format anomalies within columns. The duplicate detector finds and removes exact row duplicates. Cross-column logic checks identify redundant/mirror columns (e.g., `tipo_imposta` vs `Tipo Imposta`, `spesa` vs `SPESA TOTALE`, `cod_imposta` vs `2cod_imposta` vs `cod imposta ext`).

**Anomaly Detection**: univariate outlier detection (3σ method) flags extreme expenditure values in the `spesa` column. Categorical anomaly detection identifies rare entity descriptions and tax types.

**Remediation**: the RemediatorAgent produces a Data Reliability Score for each dataset and a detailed Human-in-the-Loop Action Plan with specific imputation strategies (e.g., median fill for numerical columns, mode fill for categorical), outlier handling recommendations, and instructions for resolving cross-column inconsistencies.

---

## Section 5 — Conclusions

AGENTS4DQ demonstrates that a multi-agent architecture powered by LLMs can automate data quality assessment on heterogeneous administrative datasets without requiring hard-coded validation rules. By combining statistical methods with LLM-driven reasoning, the system adapts to different schemas and column semantics, making it reusable across various NoiPA data sources.

The system goes beyond detection: it automatically applies high-confidence corrections (type casting, naming standardisation, duplicate removal, placeholder replacement) and delegates domain-dependent decisions to human operators through a structured Action Plan and a global Reliability Score.

**Limitations and future work**: The current pipeline runs agents sequentially: parallel execution could improve latency. The remediation step applies some fixes automatically but leaves others as suggestions; a natural extension would be letting the human operator approve or reject each suggestion interactively. Expanding support beyond CSV to JSON and database connections would better match the full range of NoiPA data sources. Additionally, incorporating more sophisticated outlier detection methods (e.g., Isolation Forest, DBSCAN) alongside the current 3σ approach could improve detection accuracy on non-normal distributions.
