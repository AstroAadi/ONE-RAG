"""
Excel Sheet Classifier
======================
Decides — per sheet and per workbook — whether content should be
processed row-wise (each row → one Qdrant point) or semantically
(markdown → chunking pipeline).

Classification technique
------------------------
A weighted score is built from three independent signals that are
each straightforward to compute from a pandas DataFrame:

  Signal 1 · numeric_col_ratio  (weight 0.40)
      Fraction of columns whose dtype is numeric (int / float).
      Financial / transactional sheets are mostly numbers.
      A sheet of explanatory text will have very few numeric columns.

  Signal 2 · short_text_score   (weight 0.30)
      1 − min(avg_str_len / 100, 1)
      Average character length of values in string columns.
      In tabular data, string cells hold short codes or names
      (e.g. "Canada", "Carretera", "Government") — typically < 20 chars.
      In prose / documentation sheets, cells hold sentences or
      bullet-point descriptions — often > 50 chars.
      Dividing by 100 and capping at 1 maps this onto [0, 1] so that
      short text → score near 1 (tabular signal) and
      long text  → score near 0 (text-heavy signal).

  Signal 3 · named_col_ratio    (weight 0.30)
      Fraction of columns with meaningful names (not "Unnamed: N").
      Real data tables always have a proper header row, so this ratio
      is 1.0 for clean transactional sheets and 0.0 for free-form
      notes sheets where pandas assigns placeholder names.

  tabular_score = 0.40 * numeric_col_ratio
                + 0.30 * short_text_score
                + 0.30 * named_col_ratio

  A sheet is classified TABULAR  when tabular_score >= TABULAR_THRESHOLD (0.55)
  A sheet is classified TEXT_HEAVY otherwise.

Hard overrides (applied before the score):
  • Fewer than MIN_ROWS (5) data rows → TEXT_HEAVY
    (not enough records to be a dataset worth row-level indexing)
  • All columns unnamed AND avg_str_len > 50 → TEXT_HEAVY
    (classic free-form notes / documentation sheet)

Workbook-level decision (classify_workbook):
  The file-level verdict is row-weighted majority: if the sheets
  classified as TABULAR contain ≥ 50 % of the total row count across
  all sheets, the workbook is routed to the row-wise pipeline;
  otherwise it goes through the semantic chunking pipeline.
  Weighting by rows (not sheet count) prevents a single large data
  sheet from being overruled by several tiny documentation sheets.
"""

from __future__ import annotations

import logging
from typing import Dict

import pandas as pd

logger = logging.getLogger(__name__)

# ── tuneable constants ────────────────────────────────────────────────────────
TABULAR_THRESHOLD   = 0.55   # score >= this → TABULAR
MIN_ROWS            = 5      # fewer rows → force TEXT_HEAVY
LONG_TEXT_THRESHOLD = 50     # avg string length above this is "long" prose
AVG_LEN_DIVISOR     = 100    # normalises avg_str_len onto [0,1]

# signal weights (must sum to 1.0)
W_NUMERIC   = 0.40
W_TEXT      = 0.30
W_NAMED_COL = 0.30


class ExcelSheetClassifier:
    """
    Classifies individual Excel sheets and whole workbooks.

    Usage
    -----
    classifier = ExcelSheetClassifier()

    # classify one sheet
    decision = classifier.classify_sheet(df)          # 'tabular' | 'text_heavy'

    # classify a whole file
    result = classifier.classify_workbook(file_path)
    # result = {
    #   'file_decision': 'row_wise' | 'semantic',
    #   'sheets': {
    #       'Sheet1': {'decision': 'tabular',    'rows': 700, 'score': 0.83},
    #       'NOTES':  {'decision': 'text_heavy', 'rows': 26,  'score': 0.41},
    #   }
    # }
    """

    # ── per-sheet classification ──────────────────────────────────────────────

    def classify_sheet(self, df: pd.DataFrame) -> str:
        """
        Return 'tabular' or 'text_heavy' for a single DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The sheet loaded by pandas (full data, no row-count cap).
        """
        # Hard override 1: too few rows to be a meaningful dataset
        if df.shape[0] < MIN_ROWS:
            logger.debug("classify_sheet → text_heavy (too few rows: %d)", df.shape[0])
            return "text_heavy"

        total_cols = max(df.shape[1], 1)

        # --- Signal 1: numeric column ratio ---
        num_cols      = df.select_dtypes(include="number").columns
        numeric_ratio = len(num_cols) / total_cols

        # --- Signal 2: average string length in text columns ---
        str_cols    = df.select_dtypes(include="object").columns
        avg_str_len = self._avg_string_length(df, str_cols)

        # --- Signal 3: named column ratio ---
        named_cols   = [c for c in df.columns if not str(c).startswith("Unnamed")]
        named_ratio  = len(named_cols) / total_cols

        # Hard override 2: all unnamed columns + long prose → documentation sheet
        if named_ratio == 0.0 and avg_str_len > LONG_TEXT_THRESHOLD:
            logger.debug(
                "classify_sheet → text_heavy (all unnamed cols, avg_str_len=%.1f)",
                avg_str_len,
            )
            return "text_heavy"

        # --- Weighted score ---
        short_text_score = 1.0 - min(avg_str_len / AVG_LEN_DIVISOR, 1.0)
        score = (
            W_NUMERIC   * numeric_ratio
            + W_TEXT    * short_text_score
            + W_NAMED_COL * named_ratio
        )

        decision = "tabular" if score >= TABULAR_THRESHOLD else "text_heavy"
        logger.debug(
            "classify_sheet → %s  (score=%.3f  numeric=%.2f  short_text=%.2f  named=%.2f)",
            decision, score, numeric_ratio, short_text_score, named_ratio,
        )
        return decision

    # ── workbook-level classification ─────────────────────────────────────────

    def classify_workbook(self, file_path: str) -> Dict:
        """
        Classify every sheet and derive a file-level routing decision.

        Parameters
        ----------
        file_path : str
            Path to an .xlsx, .xls, or .xlsm file.

        Returns
        -------
        dict with keys:
            'file_decision' : 'row_wise' | 'semantic'
            'sheets'        : dict[sheet_name → {decision, rows, score}]
        """
        xl = pd.ExcelFile(file_path)
        sheet_results: Dict = {}
        tabular_rows  = 0
        total_rows    = 0

        for sheet_name in xl.sheet_names:
            try:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
            except Exception as exc:
                logger.warning("Could not read sheet '%s': %s", sheet_name, exc)
                sheet_results[sheet_name] = {
                    "decision": "text_heavy",
                    "rows": 0,
                    "score": 0.0,
                    "error": str(exc),
                }
                continue

            decision = self.classify_sheet(df)
            score    = self._compute_score(df)   # for logging / introspection
            rows     = len(df)

            sheet_results[sheet_name] = {
                "decision": decision,
                "rows": rows,
                "score": round(score, 3),
            }

            total_rows += rows
            if decision == "tabular":
                tabular_rows += rows

            logger.info(
                "Sheet '%s': %s (score=%.3f, rows=%d)",
                sheet_name, decision.upper(), score, rows,
            )

        # Row-weighted majority vote
        tabular_fraction = tabular_rows / max(total_rows, 1)
        file_decision    = "row_wise" if tabular_fraction >= 0.50 else "semantic"

        logger.info(
            "Workbook '%s': %s  (tabular rows=%d / total=%d = %.1f%%)",
            file_path,
            file_decision.upper(),
            tabular_rows,
            total_rows,
            tabular_fraction * 100,
        )

        return {"file_decision": file_decision, "sheets": sheet_results}

    # ── private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _avg_string_length(df: pd.DataFrame, str_cols) -> float:
        """
        Mean character length of non-null string values across all
        text-typed columns.  Returns 0.0 when there are no text columns.
        """
        if len(str_cols) == 0:
            return 0.0

        total_len   = 0.0
        total_count = 0

        for col in str_cols:
            vals = df[col].dropna().astype(str)
            total_len   += vals.str.len().sum()
            total_count += len(vals)

        return total_len / max(total_count, 1)

    def _compute_score(self, df: pd.DataFrame) -> float:
        """
        Re-compute the raw score for a sheet (used for logging without
        duplicating classify_sheet's override logic).
        """
        if df.shape[0] < MIN_ROWS:
            return 0.0

        total_cols    = max(df.shape[1], 1)
        num_cols      = df.select_dtypes(include="number").columns
        str_cols      = df.select_dtypes(include="object").columns
        named_cols    = [c for c in df.columns if not str(c).startswith("Unnamed")]

        numeric_ratio    = len(num_cols) / total_cols
        avg_str_len      = self._avg_string_length(df, str_cols)
        named_ratio      = len(named_cols) / total_cols
        short_text_score = 1.0 - min(avg_str_len / AVG_LEN_DIVISOR, 1.0)

        return (
            W_NUMERIC     * numeric_ratio
            + W_TEXT      * short_text_score
            + W_NAMED_COL * named_ratio
        )
