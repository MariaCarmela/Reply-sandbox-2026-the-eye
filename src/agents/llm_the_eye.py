#====================================================================
#====================================================================
#====================================================================
# 6.  TheEye — Orchestrator
# ===========================================================================
# =======
# ===========================================================================
# 7.  TheEye — Orchestrator
# ===========================================================================
# 5.  OutputAgent
# ===========================================================================
# =======
# ===========================================================================
# 6.  OutputAgent
# ===========================================================================
# 4.  PredictionAgent
# ===========================================================================
# =======
# ===========================================================================
# 4.  LLMAgent
# ===========================================================================
class LLMAgent:
    """
    Classifies citizens using an LLM via the OpenRouter API.

    For each citizen found in personas.md, a prompt is sent to
    ``mistralai/mistral-7b-instruct:free`` asking whether the citizen
    requires preventive health support (label=1) or standard monitoring
    (label=0).

    The results are meant to *override* the ML model predictions when
    they are available.  If the API call fails for any citizen (network
    error, quota exhaustion, malformed response, etc.) the method falls
    back silently and that citizen keeps their ML-model prediction.

    Parameters
    ----------
    level : int
        Challenge level – used only for logging.
    personas_path : Path
        Path to the ``personas.md`` file for the current level.

    Attributes
    ----------
    llm_labels : dict[str, int]
        CitizenID → LLM-derived label mapping (populated after
        :py:meth:`classify_all` is called).
    """

    OPENROUTER_URL: str = "https://openrouter.ai/api/v1/chat/completions"
    MODEL: str = "mistralai/mistral-7b-instruct:free"
    PROMPT_TEMPLATE: str = (
        "You are a health risk assessment AI. Read this citizen profile and "
        "decide if they need preventive health support. Answer with ONLY '0' "
        "(standard monitoring) or '1' (activate preventive support). "
        "Citizen profile: {persona_text}"
    )

    def __init__(self, level: int, personas_path: Path) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.level = level
        self.personas_path = personas_path
        self.llm_labels: Dict[str, int] = {}

    # ------------------------------------------------------------------
    def classify_all(self) -> Dict[str, int]:
        """
        Parse personas.md, classify every citizen via the LLM, and return
        a CitizenID → label mapping.

        Citizens whose API call fails are omitted from the returned dict
        so that callers can fall back to ML predictions for them.

        Returns
        -------
        dict[str, int]
            Mapping of CitizenID → 0 or 1 as decided by the LLM.
        """
        if not _REQUESTS_AVAILABLE:
            self.logger.warning(
                "LLMAgent: 'requests' library not installed — skipping LLM stage."
            )
            return {}

        if not OPENROUTER_API_KEY:
            self.logger.warning(
                "LLMAgent: OPENROUTER_API_KEY is not set — skipping LLM stage."
            )
            return {}

        if not self.personas_path.exists():
            self.logger.warning(
                "LLMAgent: personas.md not found at %s — skipping LLM stage.",
                self.personas_path,
            )
            return {}

        citizen_blocks = self._parse_personas()
        if not citizen_blocks:
            self.logger.warning("LLMAgent: No citizen blocks parsed from personas.md.")
            return {}

        self.logger.info(
            "[Level %d] LLMAgent — classifying %d citizens via OpenRouter …",
            self.level,
            len(citizen_blocks),
        )

        results: Dict[str, int] = {}
        for cid, persona_text in citizen_blocks.items():
            label = self._classify_citizen(cid, persona_text)
            if label is not None:
                results[cid] = label

        self.llm_labels = results
        n_at_risk = sum(v for v in results.values())
        self.logger.info(
            "  LLMAgent done: %d / %d citizens classified, %d at-risk.",
            len(results),
            len(citizen_blocks),
            n_at_risk,
        )
        return results

    # ------------------------------------------------------------------
    def _parse_personas(self) -> Dict[str, str]:
        """
        Split personas.md into per-citizen text blocks.

        Returns
        -------
        dict[str, str]
            CitizenID → full block text.
        """
        text = self.personas_path.read_text(encoding="utf-8")
        blocks = re.split(r"\n(?:---+|#{1,3})\s*", text)

        citizen_id_pattern = re.compile(r"\b([A-Z0-9]{8})\b")
        citizen_blocks: Dict[str, str] = {}

        for block in blocks:
            stripped = block.strip()
            if not stripped:
                continue
            ids = citizen_id_pattern.findall(stripped)
            if not ids:
                continue
            cid = ids[0]
            citizen_blocks[cid] = stripped

        return citizen_blocks

    # ------------------------------------------------------------------
    def _classify_citizen(self, cid: str, persona_text: str) -> Optional[int]:
        """
        Send a single classification request to OpenRouter and parse the
        binary response.

        Parameters
        ----------
        cid : str
            CitizenID (used for logging only).
        persona_text : str
            The persona description block for this citizen.

        Returns
        -------
        int or None
            0 or 1 if the call succeeded and was parseable; None otherwise.
        """
        prompt = self.PROMPT_TEMPLATE.format(persona_text=persona_text)

        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/the-eye",   # optional but polite
            "X-Title": "TheEye-HealthClassifier",
        }
        payload = {
            "model": self.MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 16,
            "temperature": 0.0,
        }

        try:
            response = _requests.post(  # type: ignore[name-defined]
                self.OPENROUTER_URL,
                headers=headers,
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            raw_content: str = (
                data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                    .strip()
            )

            # Extract first digit found in the response
            digit_match = re.search(r"[01]", raw_content)
            if digit_match is None:
                self.logger.warning(
                    "  LLM[%s]: unparseable response %r — falling back to ML.",
                    cid,
                    raw_content[:80],
                )
                return None

            label = int(digit_match.group())
            risk_word = "AT-RISK" if label == 1 else "standard"
            self.logger.info(
                "  LLM[%s]: label=%d (%s) | raw=%r",
                cid,
                label,
                risk_word,
                raw_content[:120],
            )
            return label

        except Exception as exc:  # noqa: BLE001
            self.logger.warning(
                "  LLM[%s]: API call failed (%s) — falling back to ML prediction.",
                cid,
                exc,
            )
            return None


# ===========================================================================
# 5.  PredictionAgent
# ===========================================================================
# src/agents/the_eye.py
"""
The Eye - Multi-Agent AI Classification System
===============================================
Reply Sandbox 2026 Challenge

Classifies citizens as:
    0 -> Standard monitoring
    1 -> Activate preventive support

Architecture:
    - DataAgent       : Loads Status.csv, locations.json, users.json per level;
                        handles ID normalisation and manual/dynamic label assignment
    - FeatureAgent    : Engineers temporal, rolling, lag, one-hot and health features
    - GeoAgent        : Builds geospatial features from locations.json
    - PredictionAgent : Trains XGBoost with LOCO-CV; serialises model to disk
    - OutputAgent     : Writes UTF-8 plain-text submission (one CitizenID per line)
    - TheEye          : Orchestrator with optional Langfuse telemetry
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------------------------
load_dotenv()

TEAM_NAME: str = os.getenv("TEAM_NAME", "unknown_team")
LANGFUSE_SECRET_KEY: str = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_PUBLIC_KEY: str = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_HOST: str = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
 
# ---------------------------------------------------------------------------
# Path constants  (all relative to project root)
# ---------------------------------------------------------------------------
ROOT_DIR: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = ROOT_DIR / "data"
MODELS_DIR: Path = ROOT_DIR / "models"
OUTPUT_DIR: Path = ROOT_DIR / "output"

# Level names as present in the data directory
LEVEL_DIRS: Dict[int, str] = {
    1: "public_lev_1",
    2: "public_lev_2",
    3: "public_lev_3",
}

# ---------------------------------------------------------------------------
# Manual ground-truth labels for level 1 (from challenge personas)
# ---------------------------------------------------------------------------
MANUAL_LABELS_LEV1: Dict[str, int] = {
    "IAFGUHCK": 0,  # Christiane – stable
    "RFLFWVQA": 0,  # Denise – healthy
    "IXTDRHTR": 0,  # Natasha – managed
    "WNACROYX": 1,  # Craig – at risk
    "DCGGXUWF": 0,  # George – supported
}

# ---------------------------------------------------------------------------
# Keywords that indicate elevated risk (for dynamic persona parsing)
# ---------------------------------------------------------------------------
RISK_KEYWORDS: List[str] = [
    "sleep issues",
    "back pain",
    "isolation",
    "cancelling",
    "run-down",
    "irregular",
    "avoids medical",
    "energy drinks",
    "fast food",
]
# ---------------------------------------------------------------------------
# Optional requests import (used by LLMAgent)
# ---------------------------------------------------------------------------
try:
    import requests as _requests  # type: ignore

    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Optional Langfuse import (graceful degradation if not installed)
# ---------------------------------------------------------------------------
 
try:
    from langfuse import Langfuse  # type: ignore

    _LANGFUSE_AVAILABLE = True
except ImportError:
    _LANGFUSE_AVAILABLE = False


# ---------------------------------------------------------------------------
# ULID-based session ID generator (no external dependency)
# ---------------------------------------------------------------------------
def _generate_ulid() -> str:
    """Generate a ULID-like unique session identifier."""
    import time

    timestamp_ms = int(time.time() * 1000)
    random_part = os.urandom(10).hex().upper()
    return f"{timestamp_ms:013X}{random_part}"[:26]


# ===========================================================================
# 1.  DataAgent
# ===========================================================================
@dataclass
class DataAgent:
    """
    Loads and cleans all raw data sources for a given challenge level.

    Handles:
    - Status.csv  (timestamped citizen events)
    - locations.json (GPS pings per citizen)
    - users.json  (static citizen profiles)
    - BioTag / user_id → CitizenID column normalisation
    - Datetime parsing
    - Label assignment (manual for level 1, keyword-based for levels 2 & 3)

    Attributes
    ----------
    level : int
        Challenge level (1, 2, or 3).
    level_dir : Path
        Path to the data directory for this level.
    status_df : pd.DataFrame
        Cleaned status event records.
    locations_raw : list[dict]
        Raw location ping records loaded from JSON.
    users_raw : list[dict]
        Raw user profile records loaded from JSON.
    labels : dict[str, int]
        Mapping of CitizenID → binary label.
    """

    level: int
    level_dir: Path = field(init=False)
    status_df: pd.DataFrame = field(init=False, default_factory=pd.DataFrame)
    locations_raw: List[dict] = field(init=False, default_factory=list)
    users_raw: List[dict] = field(init=False, default_factory=list)
    labels: Dict[str, int] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        if self.level not in LEVEL_DIRS:
            raise ValueError(f"Level must be one of {list(LEVEL_DIRS.keys())}, got {self.level}")
        self.level_dir = DATA_DIR / LEVEL_DIRS[self.level]

    # ------------------------------------------------------------------
    def load(self) -> "DataAgent":
        """Load Status.csv, locations.json, and users.json from level_dir."""
        self.logger.info("[Level %d] Loading data from: %s", self.level, self.level_dir)

        self._load_status()
        self._load_locations()
        self._load_users()
        self._assign_labels()

        return self

    # ------------------------------------------------------------------
    def _load_status(self) -> None:
        status_path = self.level_dir / "Status.csv"
        if not status_path.exists():
            self.logger.warning("Status.csv not found at %s — using empty DataFrame.", status_path)
            self.status_df = pd.DataFrame()
            return

        df = pd.read_csv(status_path)

        # Normalise the citizen identifier column name
        df = self._normalise_citizen_id(df)

        # Parse datetime columns flexibly
        for col in df.columns:
            if "time" in col.lower() or "date" in col.lower():
                df[col] = pd.to_datetime(df[col], errors="coerce")

        # Strip whitespace from string columns
        str_cols = df.select_dtypes(include="object").columns
        df[str_cols] = df[str_cols].apply(lambda s: s.str.strip())

        df = df.drop_duplicates()
        self.status_df = df
        self.logger.info("  Status.csv loaded: %s", df.shape)

    # ------------------------------------------------------------------
    def _load_locations(self) -> None:
        loc_path = self.level_dir / "locations.json"
        if not loc_path.exists():
            self.logger.warning("locations.json not found at %s.", loc_path)
            return

        with loc_path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)

        # Accept both a bare list and a dict with a wrapping key
        if isinstance(raw, list):
            self.locations_raw = raw
        elif isinstance(raw, dict):
            # Take the first list value found
            for v in raw.values():
                if isinstance(v, list):
                    self.locations_raw = v
                    break
        self.logger.info("  locations.json loaded: %d records", len(self.locations_raw))

    # ------------------------------------------------------------------
    def _load_users(self) -> None:
        users_path = self.level_dir / "users.json"
        if not users_path.exists():
            self.logger.warning("users.json not found at %s.", users_path)
            return

        with users_path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)

        if isinstance(raw, list):
            self.users_raw = raw
        elif isinstance(raw, dict):
            for v in raw.values():
                if isinstance(v, list):
                    self.users_raw = v
                    break
        self.logger.info("  users.json loaded: %d records", len(self.users_raw))

    # ------------------------------------------------------------------
    def _normalise_citizen_id(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename BioTag / user_id / UserId variants → CitizenID."""
        rename_candidates = {
            "biotag": "CitizenID",
            "user_id": "CitizenID",
            "userid": "CitizenID",
            "citizen_id": "CitizenID",
        }
        lower_map = {c.lower(): c for c in df.columns}
        for candidate, target in rename_candidates.items():
            if candidate in lower_map and target not in df.columns:
                df = df.rename(columns={lower_map[candidate]: target})
                break
        return df

    # ------------------------------------------------------------------
    def _assign_labels(self) -> None:
        """
        Assign binary labels to citizens.

        - Level 1: use the hardcoded MANUAL_LABELS_LEV1 mapping.
        - Level 2 & 3: attempt to load labels from personas.md using
          keyword-based risk scoring; fall back to all-zeros if the file
          is missing.
        """
        if self.level == 1:
            self.labels = dict(MANUAL_LABELS_LEV1)
            self.logger.info("  Labels assigned from hardcoded map (level 1).")
            return

        # Dynamic: try personas.md in the level directory
        personas_path = self.level_dir / "personas.md"
        if not personas_path.exists():
            self.logger.warning(
                "personas.md not found at %s; defaulting all labels to 0.", personas_path
            )
            # Derive citizen IDs from whatever we loaded
            citizen_ids = self._get_all_citizen_ids()
            self.labels = {cid: 0 for cid in citizen_ids}
            return

        self.labels = self._parse_personas_md(personas_path)
        self.logger.info(
            "  Labels parsed from personas.md: %d citizens, %d at-risk.",
            len(self.labels),
            sum(self.labels.values()),
        )

    # ------------------------------------------------------------------
    def _parse_personas_md(self, path: Path) -> Dict[str, int]:
        """
        Parse personas.md to derive labels using keyword-based risk scoring.

        Expected format: sections separated by '---' or '##', each section
        contains a CitizenID (8-character uppercase alphanum) and descriptive
        text that may contain risk keywords.
        """
        text = path.read_text(encoding="utf-8")

        # Split into per-citizen blocks using markdown headings or horizontal rules
        blocks = re.split(r"\n(?:---+|#{1,3})\s*", text)

        citizen_id_pattern = re.compile(r"\b([A-Z0-9]{8})\b")
        labels: Dict[str, int] = {}

        for block in blocks:
            ids = citizen_id_pattern.findall(block)
            if not ids:
                continue
            cid = ids[0]  # First match in block is treated as the CitizenID
            block_lower = block.lower()
            risk_score = sum(1 for kw in RISK_KEYWORDS if kw in block_lower)
            labels[cid] = 1 if risk_score >= 2 else 0

        return labels

    # ------------------------------------------------------------------
    def _get_all_citizen_ids(self) -> List[str]:
        """Collect unique CitizenIDs from all loaded data sources."""
        ids: set = set()
        if not self.status_df.empty and "CitizenID" in self.status_df.columns:
            ids.update(self.status_df["CitizenID"].dropna().unique())
        for record in self.locations_raw:
            cid = record.get("CitizenID") or record.get("citizen_id") or record.get("user_id")
            if cid:
                ids.add(str(cid))
        for record in self.users_raw:
            cid = record.get("CitizenID") or record.get("citizen_id") or record.get("user_id")
            if cid:
                ids.add(str(cid))
        return list(ids)


# ===========================================================================
# 2.  FeatureAgent
# ===========================================================================
class FeatureAgent:
    """
    Builds a rich, per-citizen feature matrix from the status event log.

    Features produced
    -----------------
    Temporal
        hour, day_of_week, week_number, is_night, is_weekend
        (aggregated per citizen as mean/std of each flag)

    Health index rolling & lag features
        For PhysicalActivityIndex and SleepQualityIndex:
        rolling 3-event mean/std and lag-1 / lag-3 values
        (aggregated mean of the rolling/lag columns per citizen)

    One-hot EventType
        Mean proportion of each event type per citizen

    Health aggregates
        mean, std, min, max of PhysicalActivityIndex and SleepQualityIndex
        per citizen

    Attributes
    ----------
    status_df : pd.DataFrame
        Cleaned status records from DataAgent.
    feature_df : pd.DataFrame
        Final per-citizen feature matrix (set after build()).
    """

    def __init__(self, status_df: pd.DataFrame) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.status_df = status_df.copy() if not status_df.empty else pd.DataFrame()
        self.feature_df: pd.DataFrame = pd.DataFrame()

    # ------------------------------------------------------------------
    def build(self) -> pd.DataFrame:
        """
        Run the full feature-engineering pipeline and return the feature matrix.

        Returns
        -------
        pd.DataFrame
            One row per CitizenID with all engineered features.
        """
        if self.status_df.empty:
            self.logger.warning("status_df is empty — returning empty feature matrix.")
            self.feature_df = pd.DataFrame()
            return self.feature_df

        # Ensure CitizenID column exists
        if "CitizenID" not in self.status_df.columns:
            self.logger.error("'CitizenID' column missing from status_df.")
            self.feature_df = pd.DataFrame()
            return self.feature_df

        df = self.status_df.copy()

        # ---- 1. Temporal features ------------------------------------
        df = self._add_temporal_columns(df)

        # ---- 2. Rolling / lag health features ------------------------
        df = self._add_rolling_lag_features(df)

        # ---- 3. Per-citizen aggregation ------------------------------
        agg_parts: List[pd.DataFrame] = []

        temporal_agg = self._aggregate_temporal(df)
        if not temporal_agg.empty:
            agg_parts.append(temporal_agg)

        health_agg = self._aggregate_health(df)
        if not health_agg.empty:
            agg_parts.append(health_agg)

        rolling_agg = self._aggregate_rolling_lag(df)
        if not rolling_agg.empty:
            agg_parts.append(rolling_agg)

        event_agg = self._aggregate_event_types(df)
        if not event_agg.empty:
            agg_parts.append(event_agg)

        # ---- 4. Merge all parts --------------------------------------
        base = pd.DataFrame({"CitizenID": df["CitizenID"].unique()})
        for part in agg_parts:
            base = base.merge(part, on="CitizenID", how="left")

        # Fill any remaining NaNs with 0
        num_cols = base.select_dtypes(include="number").columns
        base[num_cols] = base[num_cols].fillna(0)

        self.feature_df = base
        self.logger.info("Feature matrix built: %s", self.feature_df.shape)
        return self.feature_df

    # ------------------------------------------------------------------
    def _add_temporal_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add hour, day_of_week, week_number, is_night, is_weekend columns."""
        ts_col = self._find_timestamp_col(df)
        if ts_col is None:
            self.logger.warning("No timestamp column found; temporal features skipped.")
            return df

        ts = pd.to_datetime(df[ts_col], errors="coerce")
        df["feat_hour"]       = ts.dt.hour
        df["feat_dow"]        = ts.dt.dayofweek          # 0=Mon … 6=Sun
        df["feat_week"]       = ts.dt.isocalendar().week.astype(int)
        df["feat_is_night"]   = ((ts.dt.hour < 6) | (ts.dt.hour >= 22)).astype(int)
        df["feat_is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
        return df

    # ------------------------------------------------------------------
    def _add_rolling_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add rolling mean/std (window=3) and lag-1 / lag-3 features for
        PhysicalActivityIndex and SleepQualityIndex per citizen.
        """
        health_cols = [
            c for c in ("PhysicalActivityIndex", "SleepQualityIndex")
            if c in df.columns
        ]
        if not health_cols:
            return df

        ts_col = self._find_timestamp_col(df)
        if ts_col:
            df = df.sort_values(["CitizenID", ts_col])
        else:
            df = df.sort_values("CitizenID")

        for col in health_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            grp = df.groupby("CitizenID")[col]
            short = col[:8]  # keep column names concise
            df[f"roll3_mean_{short}"] = grp.transform(
                lambda s: s.rolling(3, min_periods=1).mean()
            )
            df[f"roll3_std_{short}"]  = grp.transform(
                lambda s: s.rolling(3, min_periods=1).std().fillna(0)
            )
            df[f"lag1_{short}"]       = grp.transform(lambda s: s.shift(1).fillna(s.mean()))
            df[f"lag3_{short}"]       = grp.transform(lambda s: s.shift(3).fillna(s.mean()))

        return df

    # ------------------------------------------------------------------
    def _aggregate_temporal(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return per-citizen mean of temporal flag columns."""
        feat_cols = [c for c in df.columns if c.startswith("feat_")]
        if not feat_cols:
            return pd.DataFrame()

        agg = df.groupby("CitizenID")[feat_cols].mean().reset_index()
        # Rename for clarity
        agg.columns = ["CitizenID"] + [f"temporal_{c[5:]}" for c in feat_cols]
        return agg

    # ------------------------------------------------------------------
    def _aggregate_health(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return per-citizen mean/std/min/max of health index columns."""
        health_cols = [
            c for c in ("PhysicalActivityIndex", "SleepQualityIndex")
            if c in df.columns
        ]
        if not health_cols:
            return pd.DataFrame()

        records: List[Dict] = []
        for cid, grp in df.groupby("CitizenID"):
            row: Dict = {"CitizenID": cid}
            for col in health_cols:
                vals = pd.to_numeric(grp[col], errors="coerce").dropna()
                short = col[:8]
                row[f"health_mean_{short}"] = vals.mean() if not vals.empty else 0.0
                row[f"health_std_{short}"]  = vals.std()  if len(vals) > 1 else 0.0
                row[f"health_min_{short}"]  = vals.min()  if not vals.empty else 0.0
                row[f"health_max_{short}"]  = vals.max()  if not vals.empty else 0.0
            records.append(row)

        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    def _aggregate_rolling_lag(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return per-citizen mean of all rolling/lag feature columns."""
        rl_cols = [
            c for c in df.columns
            if c.startswith("roll3_") or c.startswith("lag")
        ]
        if not rl_cols:
            return pd.DataFrame()

        agg = df.groupby("CitizenID")[rl_cols].mean().reset_index()
        return agg

    # ------------------------------------------------------------------
    def _aggregate_event_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode EventType and return per-citizen mean proportions."""
        if "EventType" not in df.columns:
            return pd.DataFrame()

        dummies = pd.get_dummies(df["EventType"].astype(str), prefix="event")
        tmp = pd.concat([df[["CitizenID"]], dummies], axis=1)
        agg = tmp.groupby("CitizenID").mean().reset_index()
        return agg

    # ------------------------------------------------------------------
    @staticmethod
    def _find_timestamp_col(df: pd.DataFrame) -> Optional[str]:
        """Return the name of the first column that looks like a timestamp."""
        for col in df.columns:
            if "time" in col.lower() or "date" in col.lower():
                return col
        # Fallback: check dtype
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                return col
        return None


# ===========================================================================
# 3.  GeoAgent
# ===========================================================================
class GeoAgent:
    """
    Builds geospatial features from the raw locations.json records.

    Features produced (per CitizenID)
    ----------------------------------
    geo_event_count         : total location pings
    geo_mean_lat            : mean latitude
    geo_mean_lng            : mean longitude
    geo_std_lat             : std of latitude
    geo_std_lng             : std of longitude
    geo_mean_dist_centroid  : mean Haversine distance (km) from city centroid
    geo_max_dist_centroid   : max  Haversine distance (km) from city centroid
    geo_unique_cities       : number of distinct cities visited

    Attributes
    ----------
    locations_raw : list[dict]
        Raw location records from DataAgent.
    geo_df : pd.DataFrame
        Per-citizen geospatial feature DataFrame (set after build()).
    """

    EARTH_RADIUS_KM: float = 6371.0

    def __init__(self, locations_raw: List[dict]) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.locations_raw = locations_raw
        self.geo_df: pd.DataFrame = pd.DataFrame()

    # ------------------------------------------------------------------
    def build(self) -> pd.DataFrame:
        """
        Process raw location records and return the geo feature matrix.

        Returns
        -------
        pd.DataFrame
            One row per CitizenID with all geospatial features.
        """
        if not self.locations_raw:
            self.logger.warning("No location records available; returning empty geo features.")
            return pd.DataFrame()

        df = pd.DataFrame(self.locations_raw)

        # Normalise column names
        df = self._normalise_columns(df)

        if "CitizenID" not in df.columns:
            self.logger.warning("CitizenID column missing from locations data.")
            return pd.DataFrame()

        # Coerce lat/lng to numeric
        for col in ("latitude", "longitude"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["latitude", "longitude"])
        if df.empty:
            self.logger.warning("All location records have invalid lat/lng.")
            return pd.DataFrame()

        # ---- City centroids -----------------------------------------
        city_col = self._find_city_col(df)
        if city_col:
            centroids = (
                df.groupby(city_col)[["latitude", "longitude"]]
                  .mean()
                  .rename(columns={"latitude": "centroid_lat", "longitude": "centroid_lng"})
            )
            df = df.merge(centroids, on=city_col, how="left")
        else:
            # Global centroid as fallback
            df["centroid_lat"] = df["latitude"].mean()
            df["centroid_lng"] = df["longitude"].mean()

        # ---- Haversine distance from centroid ------------------------
        df["dist_centroid_km"] = df.apply(
            lambda row: self._haversine(
                row["latitude"], row["longitude"],
                row["centroid_lat"], row["centroid_lng"],
            ),
            axis=1,
        )

        # ---- Per-citizen aggregation --------------------------------
        records: List[Dict] = []
        for cid, grp in df.groupby("CitizenID"):
            row: Dict = {
                "CitizenID":              cid,
                "geo_event_count":        len(grp),
                "geo_mean_lat":           grp["latitude"].mean(),
                "geo_mean_lng":           grp["longitude"].mean(),
                "geo_std_lat":            grp["latitude"].std()  if len(grp) > 1 else 0.0,
                "geo_std_lng":            grp["longitude"].std() if len(grp) > 1 else 0.0,
                "geo_mean_dist_centroid": grp["dist_centroid_km"].mean(),
                "geo_max_dist_centroid":  grp["dist_centroid_km"].max(),
                "geo_unique_cities":      grp[city_col].nunique() if city_col else 0,
            }
            records.append(row)

        self.geo_df = pd.DataFrame(records).fillna(0)
        self.logger.info("Geo feature matrix built: %s", self.geo_df.shape)
        return self.geo_df

    # ------------------------------------------------------------------
    def _normalise_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardise column names to expected schema."""
        rename: Dict[str, str] = {}
        lower_map = {c.lower(): c for c in df.columns}

        id_candidates = ["biotag", "user_id", "userid", "citizen_id", "citizenid"]
        for cand in id_candidates:
            if cand in lower_map and "CitizenID" not in df.columns:
                rename[lower_map[cand]] = "CitizenID"
                break

        for lat_cand in ("lat", "latitude"):
            if lat_cand in lower_map and "latitude" not in df.columns:
                rename[lower_map[lat_cand]] = "latitude"
                break

        for lng_cand in ("lng", "lon", "longitude"):
            if lng_cand in lower_map and "longitude" not in df.columns:
                rename[lower_map[lng_cand]] = "longitude"
                break

        if rename:
            df = df.rename(columns=rename)
        return df

    # ------------------------------------------------------------------
    @staticmethod
    def _find_city_col(df: pd.DataFrame) -> Optional[str]:
        """Return the name of the city column if present."""
        for col in df.columns:
            if col.lower() in ("city", "town", "zone", "district", "area"):
                return col
        return None

    # ------------------------------------------------------------------
    def _haversine(
        self,
        lat1: float, lng1: float,
        lat2: float, lng2: float,
    ) -> float:
        """Return the Haversine great-circle distance in kilometres."""
        R = self.EARTH_RADIUS_KM
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi       = np.radians(lat2 - lat1)
        dlambda    = np.radians(lng2 - lng1)
        a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
        return float(2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))


# ===========================================================================
# 4.  PredictionAgent
# ===========================================================================
class PredictionAgent:
    """
    Trains an XGBoost binary classifier and generates predictions.

    Strategy
    --------
    - For datasets with >= 5 citizens: Leave-One-Citizen-Out (LOCO) cross-
      validation is used to estimate generalisation performance.
    - Final model is trained on ALL available labelled citizens.
    - The trained model is persisted to models/xgb_lev{level}.pkl.

    Attributes
    ----------
    level : int
        Challenge level (used for model file naming).
    model : XGBClassifier
        Trained XGBoost model.
    feature_cols : list[str]
        Ordered feature column names used during training.
    """

    def __init__(self, level: int, random_state: int = 42, **xgb_kwargs) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.level = level
        self.random_state = random_state
        self.feature_cols: List[str] = []
        self.model: Optional[XGBClassifier] = None

        default_params: Dict = dict(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=5,        # compensate class imbalance
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        )
        default_params.update(xgb_kwargs)
        self._xgb_params = default_params

    # ------------------------------------------------------------------
    def train(
        self,
        feature_df: pd.DataFrame,
        labels: Dict[str, int],
    ) -> "PredictionAgent":
        """
        Train the XGBoost classifier.

        Parameters
        ----------
        feature_df : pd.DataFrame
            Per-citizen feature matrix (must contain 'CitizenID' column).
        labels : dict[str, int]
            CitizenID → label mapping.

        Returns
        -------
        PredictionAgent
            Self, for method chaining.
        """
        if feature_df.empty:
            self.logger.error("Feature matrix is empty — cannot train.")
            return self

        # Attach labels
        df = feature_df.copy()
        df["label"] = df["CitizenID"].map(labels)
        labelled = df.dropna(subset=["label"])

        if labelled.empty:
            self.logger.error("No labelled citizens found — cannot train.")
            return self

        non_feature = {"CitizenID", "label"}
        self.feature_cols = [c for c in labelled.columns if c not in non_feature]

        X = labelled[self.feature_cols].astype(float)
        y = labelled["label"].astype(int)
        groups = labelled["CitizenID"].values

        self.logger.info(
            "[Level %d] Training on %d citizens (%d positive, %d negative).",
            self.level, len(y), int(y.sum()), int((y == 0).sum()),
        )

        # ---- LOCO cross-validation (when enough citizens) -----------
        if len(y) >= 5:
            self._loco_cv(X, y, groups)
        else:
            self.logger.warning(
                "Only %d citizens — skipping LOCO CV (need >= 5).", len(y)
            )

        # ---- Train final model on ALL labelled data -----------------
        self.model = XGBClassifier(**self._xgb_params)
        self.model.fit(X, y, verbose=False)
        train_f1 = f1_score(y, self.model.predict(X), average="macro", zero_division=0)
        self.logger.info("  Final model — Train Macro-F1: %.4f", train_f1)

        # Persist model
        self._save_model()

        return self

    # ------------------------------------------------------------------
    def predict(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate binary predictions for all citizens in *feature_df*.

        Parameters
        ----------
        feature_df : pd.DataFrame
            Per-citizen feature matrix.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ['CitizenID', 'label'].
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained. Call train() first.")

        if feature_df.empty:
            return pd.DataFrame(columns=["CitizenID", "label"])

        # Align to training feature columns, filling missing with 0
        X = (
            feature_df.reindex(columns=self.feature_cols, fill_value=0)
                      .astype(float)
        )
        preds = self.model.predict(X)

        result = pd.DataFrame({
            "CitizenID": feature_df["CitizenID"].values,
            "label":     preds.astype(int),
        })
        n_flagged = int((result["label"] == 1).sum())
        self.logger.info(
            "  Predictions: %d / %d citizens flagged (label=1).",
            n_flagged, len(result),
        )
        return result

    # ------------------------------------------------------------------
    def _loco_cv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        groups: np.ndarray,
    ) -> None:
        """Run Leave-One-Citizen-Out CV and log macro-F1 scores."""
        logo = LeaveOneGroupOut()
        scores: List[float] = []

        for train_idx, test_idx in logo.split(X, y, groups):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

            # Skip folds where training set has only one class
            if len(np.unique(y_tr)) < 2:
                continue

            fold_model = XGBClassifier(**self._xgb_params)
            fold_model.fit(X_tr, y_tr, verbose=False)
            fold_pred  = fold_model.predict(X_te)
            fold_f1    = f1_score(y_te, fold_pred, average="macro", zero_division=0)
            scores.append(fold_f1)

        if scores:
            self.logger.info(
                "  LOCO-CV Macro-F1: %.4f ± %.4f (n=%d folds)",
                np.mean(scores), np.std(scores), len(scores),
            )
        else:
            self.logger.warning("  LOCO-CV: no valid folds produced.")

    # ------------------------------------------------------------------
    def _save_model(self) -> None:
        """Persist the trained model to models/xgb_lev{level}.pkl."""
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / f"xgb_lev{self.level}.pkl"
        with model_path.open("wb") as fh:
            pickle.dump(
                {"model": self.model, "feature_cols": self.feature_cols},
                fh,
            )
        self.logger.info("  Model saved to: %s", model_path.resolve())


# ===========================================================================
# 5.  OutputAgent
# ===========================================================================
class OutputAgent:
    """
    Formats predictions into the challenge-required submission file.

    Output format
    -------------
    A plain UTF-8 text file (output/submission_lev{N}.txt) containing
    exactly one CitizenID per line for every citizen predicted as label=1.
    Citizens are sorted lexicographically.

    Attributes
    ----------
    level : int
        Challenge level (used for file naming).
    predictions_df : pd.DataFrame
        DataFrame with columns ['CitizenID', 'label'].
    """

    def __init__(self, level: int, predictions_df: pd.DataFrame) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.level = level
        self.predictions_df = predictions_df.copy()

    # ------------------------------------------------------------------
    def save(self, path: Optional[Path] = None) -> Path:
        """
        Write the submission file.

        Parameters
        ----------
        path : Path, optional
            Override output path. Defaults to
            output/submission_lev{level}.txt.

        Returns
        -------
        Path
            Resolved path to the written file.
        """
        if path is None:
            path = OUTPUT_DIR / f"submission_lev{self.level}.txt"

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        flagged = (
            self.predictions_df[self.predictions_df["label"] == 1]["CitizenID"]
            .sort_values()
            .tolist()
        )

        with path.open("w", encoding="utf-8") as fh:
            for cid in flagged:
                fh.write(f"{cid}\n")

        self.logger.info(
            "[Level %d] Submission saved: %d at-risk citizens → %s",
            self.level, len(flagged), path.resolve(),
        )
        return path.resolve()


# ===========================================================================
# 6.  TheEye — Orchestrator
# ===========================================================================
class TheEye:
    """
    Top-level orchestrator that coordinates all agents for a given level.

    Pipeline per level
    ------------------
    1. DataAgent    : load all sources, assign labels
    2. FeatureAgent : build temporal / health / event feature matrix
    3. GeoAgent     : build geospatial feature matrix
    4. Merge        : join Feature + Geo matrices on CitizenID
    5. PredictionAgent : LOCO-CV evaluation + final training + inference
    6. OutputAgent  : write submission file

    Optional Langfuse telemetry is enabled when LANGFUSE_SECRET_KEY is set
    in the environment.

    Parameters
    ----------
    level : int
        Challenge level to process (1, 2, or 3).
    """

    def __init__(self, level: int) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.level  = level
        # Agent references (populated during run)
        self.data_agent:       Optional[DataAgent]       = None
        self.feature_agent:    Optional[FeatureAgent]    = None
        self.geo_agent:        Optional[GeoAgent]        = None
        self.prediction_agent: Optional[PredictionAgent] = None
        self.llm_agent:        Optional[LLMAgent]        = None
        self.output_agent:     Optional[OutputAgent]     = None
 

        # Langfuse client (None if unavailable / unconfigured)
        self._langfuse: Optional[object] = self._init_langfuse()
        self._session_id: str = _generate_ulid()

    # ------------------------------------------------------------------
    def run(self) -> Tuple[Path, pd.DataFrame]:
        """
        Execute the full pipeline for this level.

        Returns
        -------
        tuple[Path, pd.DataFrame]
            (submission_file_path, predictions_dataframe)
        """
        self.logger.info("=" * 65)
        self.logger.info(
            "  THE EYE | Level %d | Session %s | Team: %s",
            self.level, self._session_id, TEAM_NAME,
        )
        self.logger.info("=" * 65)

        self._trace("pipeline_start", {"level": self.level, "team": TEAM_NAME})

        # ----------------------------------------------------------------
        # Stage 1 : Data ingestion
        # ----------------------------------------------------------------
        self.logger.info("[Stage 1/5] DataAgent — loading data …")
        self.data_agent = DataAgent(level=self.level)
        self.data_agent.load()
        self._trace("data_loaded", {"status_rows": len(self.data_agent.status_df)})

        # ----------------------------------------------------------------
        # Stage 2 : Feature engineering
        # ----------------------------------------------------------------
        self.logger.info("[Stage 2/5] FeatureAgent — engineering features …")
        self.feature_agent = FeatureAgent(status_df=self.data_agent.status_df)
        feature_df = self.feature_agent.build()

        # ----------------------------------------------------------------
        # Stage 3 : Geospatial features
        # ----------------------------------------------------------------
        self.logger.info("[Stage 3/5] GeoAgent — building geo features …")
        self.geo_agent = GeoAgent(locations_raw=self.data_agent.locations_raw)
        geo_df = self.geo_agent.build()

        # ---- Merge Feature + Geo matrices ----------------------------
        if not feature_df.empty and not geo_df.empty:
            combined_df = feature_df.merge(geo_df, on="CitizenID", how="outer")
        elif not feature_df.empty:
            combined_df = feature_df
        elif not geo_df.empty:
            combined_df = geo_df
        else:
            self.logger.error("Both feature matrices are empty — cannot proceed.")
            raise RuntimeError("Feature engineering produced no data.")

        # Fill NaNs introduced by outer join
        num_cols = combined_df.select_dtypes(include="number").columns
        combined_df[num_cols] = combined_df[num_cols].fillna(0)

        self.logger.info("  Combined feature matrix: %s", combined_df.shape)
        self._trace("features_ready", {"shape": str(combined_df.shape)})

        # ----------------------------------------------------------------
        # Stage 4 : Training & prediction
        # ----------------------------------------------------------------
        self.logger.info("[Stage 4/6] PredictionAgent — train & predict …")
        self.prediction_agent = PredictionAgent(level=self.level)
        self.prediction_agent.train(
            feature_df=combined_df,
            labels=self.data_agent.labels,
        )
        predictions_df = self.prediction_agent.predict(combined_df)
        self._trace(
            "predictions_done",
            {
                "n_citizens":  len(predictions_df),
                "n_at_risk":   int((predictions_df["label"] == 1).sum()),
            },
        )

        # ----------------------------------------------------------------
        # Stage 4.5 : LLM override (only when personas.md exists)
        # ----------------------------------------------------------------
        personas_path = self.data_agent.level_dir / "personas.md"
        if personas_path.exists():
            self.logger.info("[Stage 4.5/6] LLMAgent — overriding with LLM classifications …")
            self.llm_agent = LLMAgent(level=self.level, personas_path=personas_path)
            llm_labels = self.llm_agent.classify_all()

            if llm_labels:
                # Apply LLM overrides row-by-row; keep ML prediction where
                # the LLM returned no result (API failure / missing citizen).
                original_counts = predictions_df["label"].value_counts().to_dict()
                predictions_df["label"] = predictions_df.apply(
                    lambda row: llm_labels.get(row["CitizenID"], row["label"]),
                    axis=1,
                )
                overridden = sum(
                    1 for cid in predictions_df["CitizenID"]
                    if cid in llm_labels
                )
                new_counts = predictions_df["label"].value_counts().to_dict()
                self.logger.info(
                    "  LLM overrides applied: %d citizens updated. "
                    "Label distribution before=%s after=%s",
                    overridden,
                    original_counts,
                    new_counts,
                )
                self._trace(
                    "llm_override_done",
                    {
                        "n_overridden": overridden,
                        "n_at_risk_after": int((predictions_df["label"] == 1).sum()),
                    },
                )
            else:
                self.logger.info(
                    "  LLMAgent returned no labels — ML predictions unchanged."
                )
        else:
            self.logger.info(
                "[Stage 4.5/6] LLMAgent — personas.md not found, skipping LLM stage."
            )

        # ----------------------------------------------------------------
        # Stage 5 : Output
        # ----------------------------------------------------------------
        self.logger.info("[Stage 5/6] OutputAgent — saving submission …")
        self.output_agent = OutputAgent(
            level=self.level,
            predictions_df=predictions_df,
        )
        submission_path = self.output_agent.save()
        self._trace("submission_saved", {"path": str(submission_path)})

        self.logger.info("=" * 65)
        self.logger.info(
            "  Level %d complete → %s", self.level, submission_path
        )
        self.logger.info("=" * 65)

        return submission_path, predictions_df

    # ------------------------------------------------------------------
    def _init_langfuse(self) -> Optional[object]:
        """Initialise Langfuse client if credentials are configured."""
        if not _LANGFUSE_AVAILABLE:
            return None
        if not LANGFUSE_SECRET_KEY or not LANGFUSE_PUBLIC_KEY:
            return None
        try:
            client = Langfuse(  # type: ignore[name-defined]
                secret_key=LANGFUSE_SECRET_KEY,
                public_key=LANGFUSE_PUBLIC_KEY,
                host=LANGFUSE_HOST,
            )
            self.logger.info("Langfuse telemetry enabled.")
            return client
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("Could not initialise Langfuse: %s", exc)
            return None

    # ------------------------------------------------------------------
    def _trace(self, event_name: str, metadata: Optional[Dict] = None) -> None:
        """
        Emit a Langfuse trace event if the client is available.
        Silently no-ops when Langfuse is not configured.
        """
        if self._langfuse is None:
            return
        try:
            self._langfuse.trace(  # type: ignore[attr-defined]
                name=event_name,
                session_id=self._session_id,
                metadata=metadata or {},
                tags=[f"level_{self.level}", TEAM_NAME],
            )
        except Exception as exc:  # noqa: BLE001
            self.logger.debug("Langfuse trace failed (non-fatal): %s", exc)


# ===========================================================================
# Entry point
# ===========================================================================
def main() -> None:
    """
    Run TheEye on all three challenge levels sequentially and print a
    summary table.
    """
    results: List[Dict] = []
    all_levels = [1, 2, 3]

    for level in all_levels:
        try:
            eye = TheEye(level=level)
            submission_path, predictions_df = eye.run()

            n_total   = len(predictions_df)
            n_at_risk = int((predictions_df["label"] == 1).sum())
            results.append(
                {
                    "Level":       level,
                    "Citizens":    n_total,
                    "At-Risk (1)": n_at_risk,
                    "Safe (0)":    n_total - n_at_risk,
                    "File":        str(submission_path),
                }
            )
        except Exception as exc:  # noqa: BLE001
            logging.getLogger("main").error(
                "Level %d failed: %s", level, exc, exc_info=True
            )
            results.append(
                {
                    "Level":       level,
                    "Citizens":    0,
                    "At-Risk (1)": 0,
                    "Safe (0)":    0,
                    "File":        f"ERROR: {exc}",
                }
            )

    # ---- Summary table -----------------------------------------------
    print("\n" + "=" * 70)
    print("  THE EYE — Run Summary")
    print("=" * 70)
    header = f"{'Level':>6}  {'Citizens':>10}  {'At-Risk':>9}  {'Safe':>8}  File"
    print(header)
    print("-" * 70)
    for row in results:
        print(
            f"{row['Level']:>6}  "
            f"{row['Citizens']:>10}  "
            f"{row['At-Risk (1)']:>9}  "
            f"{row['Safe (0)']:>8}  "
            f"{row['File']}"
        )
    print("=" * 70 + "\n")


if __name__ == "__main__":    main()