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

1. **Validates** the schema (data types and naming conventions)
2. **Analyzes completeness** (null/missing values, placeholder detection, sparse columns)
3. **Checks consistency** (duplicate detection, format uniformity, cross-column logical rules)
4. **Detects anomalies** (univariate statistical outliers and rare categorical values)
5. **Suggests remediations** for each issue found

The system produces a structured quality report and presents it through a **Streamlit** web interface with a typewriter-style streaming output.

---

## Section 2 — Methods

### 2.1 Architecture

AGENTS4DQ follows a **Supervisor architecture** where a central orchestrator (`DataOrchestrator`) receives the user's request and coordinates four specialized agents in sequence:

```
User (CSV upload + prompt)
        │
        ▼
┌─────────────────────┐
│  DataOrchestrator    │  ← Intent detection + CSV loading
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  SchemaValidator     │  ← Step 1: Data type + naming convention checks
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  CompletenessAnalyst │  ← Step 2: Missing values, placeholders, sparse columns
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  ConsistencyValidator│  ← Step 3: Duplicates, format consistency, cross-column logic
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  AnomalyDetector     │  ← Step 4: Univariate outliers + categorical anomalies
└────────┬────────────┘
         │
         ▼
   Streamlit Report
```

### 2.2 Agent Details

| Agent | LLM Backend | Key Responsibilities |
|---|---|---|
| **DataOrchestrator** | StepFun Step-3.5-Flash (via OpenRouter) | Parses user intent; loads CSV/XLSX into a DataFrame |
| **SchemaValidator** | StepFun Step-3.5-Flash (via OpenRouter) | Checks column data types against logical expectations; validates naming conventions |
| **CompletenessAnalyst** | StepFun Step-3.5-Flash (via OpenRouter) | Detects placeholder strings (e.g. "N/A", "-", "null") via LLM; computes per-column and overall missing percentages; flags sparse columns (>50% missing) for removal |
| **ConsistencyValidator** | Google Gemini 3.1 Flash Lite Preview | Identifies key columns and detects duplicates; uses regex pattern analysis to flag format inconsistencies; validates cross-column logical relationships on a sample |
| **AnomalyDetector** | StepFun Step-3.5-Flash (via OpenRouter) | LLM selects candidate numerical columns → 3σ outlier detection; LLM selects candidate categorical columns → flags values appearing in <1% of rows |

### 2.3 Design Choices

- **LLM-driven column selection**: Rather than hard-coding which columns to validate, the agents ask the LLM to infer which columns are relevant for each check (e.g., which columns are numerical candidates for outlier detection). This makes the system **dataset-agnostic**.
- **Regex pattern fingerprinting** (`functions.py`): Each cell value is converted into a pattern string (digits → `N`, letters → `W`, punctuation preserved) to build a frequency map per column. This enables format consistency detection without manually defining expected formats.
- **Hybrid approach**: Statistical checks (3σ for outliers, frequency thresholds for categorical anomalies, exact duplicate detection) are combined with LLM reasoning (placeholder identification, cross-column logic, naming conventions).
- **SQLite caching** (`langchain` `SQLiteCache`): LLM responses are cached to avoid redundant API calls during development and demo.
- **Streamlit frontend**: Results are streamed with a typewriter effect for a more readable, step-by-step presentation.

### 2.4 Environment Setup

```bash
# Clone the repository
git clone https://github.com/<org>/AGENTS4DQ-807861-vecchia.git
cd AGENTS4DQ-807861-vecchia

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

A `.env` file is required at the project root with the following keys:

```
OPENROUTER_API_KEY=<your_openrouter_key>
GOOGLE_API_KEY=<your_google_genai_key>
```

To launch the application:

```bash
cd src
streamlit run main.py
```

---

## Section 3 — Experimental Design

### Experiment 1 — End-to-end Quality Assessment on `spesa.csv`

- **Purpose**: Validate that the full agent pipeline correctly identifies schema issues, missing data, inconsistencies, and outliers on a real NoiPA expenditure dataset (18 columns, including intentionally noisy/redundant columns such as `SPESA TOTALE`, `2cod_imposta`, `cod imposta ext`, `ente%code`).
- **Baseline**: Manual inspection of the dataset in a spreadsheet editor.
- **Evaluation Metrics**: Qualitative comparison — whether each agent detects the expected issues (e.g., redundant columns, naming convention violations, type mismatches, format inconsistencies in the pattern report).

### Experiment 2 — End-to-end Quality Assessment on `attivazioniCessazioni.csv`

- **Purpose**: Test generalization to a structurally different NoiPA dataset (personnel activations/terminations, with columns like `mese`, `anno`, `provincia_sede`, `attivazioni`, `cessazioni`, `qualifica`, and intentionally noisy columns such as `3descrizione`, `regione%sede`, `att ivazioni`).
- **Baseline**: Manual inspection.
- **Evaluation Metrics**: Same qualitative assessment — the system should flag naming violations (`3descrizione`, `regione%sede`, `att ivazioni`), detect cross-column logic (e.g., `mese`/`anno` coherence with `RATA`), spot province casing inconsistencies (e.g., `Aq` vs `AQ`), and identify redundant mirror columns.

---

## Section 4 — Results

> **Note:** Figures below are generated from code output (Streamlit screenshots). See the `images/` folder.

### Key Findings

**Schema Validation**: The SchemaValidator correctly identifies type mismatches (e.g., identifier columns stored as float/int instead of string) and naming violations (columns with special characters like `%`, leading digits, or spaces such as `ente%code`, `3descrizione`, `att ivazioni`).

**Completeness Analysis**: The CompletenessAnalyst detects placeholder values across both datasets and computes per-column missing rates. Columns like `note` and `fonte_dato` are flagged as largely empty and candidates for removal.

**Consistency Validation**: The regex pattern fingerprinting approach proves effective at detecting format anomalies within columns. Cross-column logic checks identify redundant/mirror columns (e.g., `tipo_imposta` vs `Tipo Imposta`, `spesa` vs `SPESA TOTALE`, `cod_imposta` vs `2cod_imposta` vs `cod imposta ext`).

**Anomaly Detection**: Univariate outlier detection (3σ method) flags extreme expenditure values in the `spesa` column. Categorical anomaly detection identifies rare entity descriptions and tax types.

<!-- TODO: Add Streamlit screenshot figures here -->
<!-- ![Schema Validation Output](images/schema_validation.png) -->
<!-- ![Completeness Report](images/completeness_report.png) -->
<!-- ![Consistency Report](images/consistency_report.png) -->
<!-- ![Anomaly Detection Output](images/anomaly_detection.png) -->

---

## Section 5 — Conclusions

AGENTS4DQ demonstrates that a multi-agent architecture powered by LLMs can automate data quality assessment on heterogeneous administrative datasets without requiring hard-coded validation rules. By combining statistical methods with LLM-driven reasoning, the system adapts to different schemas and column semantics, making it reusable across various NoiPA data sources.

**Limitations and future work**: The current pipeline runs agents sequentially — parallel execution could improve latency. The remediation step is currently limited to suggestions within the agent outputs rather than producing a corrected dataset file. A natural extension would be adding a dedicated **Remediation Agent** that applies high-confidence fixes and exports a cleaned CSV alongside the quality report. Additionally, incorporating a global **Reliability Score** (0–100) that aggregates findings across all four dimensions would provide a single summary metric for data stakeholders. Finally, expanding support beyond CSV to JSON and database connections would better match the full range of NoiPA data sources.