import pandas as pd
import re
from datetime import datetime
import numpy as np
from typing import Callable, Optional, Any
from sklearn.preprocessing import StandardScaler

from preprocessing.data_completion import DataComplete, DefaultValue
from preprocessing.conversions import (
    Columns, month_prefix, equiv_low, equiv_mid, equiv_high,
    equiv_neg, equiv_pos
)

MISSING_VALUE_DEFAULT = np.nan


def to_lower(val: str) -> str:
    return str(val).lower()


def to_upper(val: str) -> str:
    return str(val).upper()


def extract_percent(s):
    match = re.search(r'(\d+(?:\.\d+)?)\s*%', s)
    return float(match.group(1)) if match else None


def extract_percent_range(s):
    match = re.search(r'(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*%', s)
    if match:
        x = float(match.group(1))
        y = float(match.group(2))
        return x, y
    return None


def extract_first_number(s):
    match = re.search(r'\d+(?:\.\d+)?', s)
    return float(match.group()) if match else None


def replace_contains(rules: list[tuple[str, int]],
                     strong: bool = False,
                     mapper: Optional[Callable] = None,
                     default: Any = MISSING_VALUE_DEFAULT) -> Callable:
    def inner(val: str):
        if not val or pd.isna(val):
            return default
        if mapper is not None:
            val = mapper(val)
        if strong:
            val = re.split(r'\s*[+/\-]\s*', val)
        for mark, rank in rules:
            if mark in val:
                return rank
        return default
    return inner


def filter_her2(val: str) -> int:
    """
    FISH - worse
    IHC - less worse
    higher percentage - bad
    """
    val = str(val).lower()

    num = 0
    if "+1" in val:
        num = 1
    if "+2" in val:
        num = 2
    if "+3" in val:
        num = 3

    status = None
    if equiv_pos(val):
        status = True
    if equiv_neg(val):
        status = False

    type_ = None
    if "ihc" in val:
        type_ = 1
    if "fish" in val:
        type_ = 2

    percent = extract_percent(val)
    res = 0
    res += num
    if type_ == 1:
        res += 10
    if type_ == 2:
        res += 30
    if percent is not None:
        res += percent / 25
    if status:
        res += 40
    if not status and status is not None:
        res *= -1

    return res


def filter_k167(val: str) -> float:
    val = str(val).lower()

    # ignore if contains month name
    if any(month in val for month in month_prefix):
        return 0
    r = extract_percent_range(val)
    if r is not None:
        x, y = r
        return (x + y) / 2
    # search for score
    if "score" in val:
        num = extract_first_number(val)
        if num is not None:
            return num * 20

    num = extract_first_number(val)
    if num is not None:
        # assume ment to be percentage, normalize to [0, 100] if needed
        while num > 100:
            num /= 10

        return num

    # TODO - validate low, mid, high values against averages of valid data of column
    if equiv_low(val):
        return 5
    if equiv_mid(val):
        return 40
    if equiv_high(val):
        return 80

    if equiv_neg(val):
        return 5
    if equiv_pos(val):
        return 80

    return 0


def filter_histological_diagnosis(val):
    val = str(val).upper()

    if any(keyword in val for keyword in [
        "BENIGN", "ADENOMA", "FIBROADENOMA", "PAPILLOMA", "PAPILLOMATOSIS"
    ]):
        return 0  # Benign

    elif "IN SITU" in val or any(keyword in val for keyword in [
        "DCIS", "LCIS", "INTRADUCTAL", "LOBULAR CARCINOMA IN SITU",
        "CARCINOMA IN SITU"
    ]):
        return 1  # In situ carcinoma

    elif any(keyword in val for keyword in [
        "INFILTRATING", "INVASIVE", "COMEDO", "DUCTAL", "LOBULAR", "TUBULAR",
        "MEDULLARY",
        "MUCIN", "APOCRINE", "PAGET", "PHYLLODES"
    ]):
        return 2  # Invasive carcinoma

    elif any(keyword in val for keyword in [
        "NEUROENDOCRINE", "INFLAMMATORY", "MALIGNANT", "NOS", "CARCINOMA"
    ]):
        return 3  # Aggressive or unspecified carcinoma

    else:
        return 3  # Default to most severe if unrecognized


def filter_er(val: str):
    val = str(val).lower()
    percent = extract_percent(val)
    if equiv_neg(val) or (percent is not None and percent == 0):
        return 0
    if equiv_pos(val) or (percent is not None and percent >= 90) or "+3" in val:
        return 3
    if equiv_mid(val) or (percent is not None and percent >= 50) or "+2" in val or "2+" in val:
        return 2
    if (percent is not None and percent > 0) or "+1" in val or "1+" in val:
        return 1

    return MISSING_VALUE_DEFAULT


def filter_pr(val: str):
    val = str(val).lower()
    percent = extract_percent(val)
    if equiv_neg(val) or (percent is not None and percent == 0):
        return 0
    if equiv_pos(val) or (percent is not None and percent >= 90) or "+3" in val:
        return 3
    if equiv_mid(val) or (percent is not None and percent >= 50) or "+2" in val or "2+" in val:
        return 2
    if (percent is not None and percent > 0) or "+1" in val or "1+" in val:
        return 1

    return MISSING_VALUE_DEFAULT


def filter_nodes_exam(val) -> float:
    try:
        # טיפוס מספרי תקני
        if isinstance(val, (int, float)):
            return float(val)

        # נסה לחלץ מספר מהמחרוזת
        num = extract_first_number(val)
        return num if num is not None else MISSING_VALUE_DEFAULT

    except Exception:
        return MISSING_VALUE_DEFAULT


def filter_tnm_n(val: str) -> int:
    if not isinstance(val, str):
        return -1

    val = val.strip().upper()

    if 'N0' in val or 'ITC' in val:
        return 0
    elif 'N1' in val:
        return 1
    elif 'N2' in val:
        return 2
    elif 'N3' in val or 'N4' in val:
        return 3
    elif 'NX' in val or 'NOT YET ESTABLISHED' in val:
        return MISSING_VALUE_DEFAULT

    return MISSING_VALUE_DEFAULT


def filter_positive_nodes(val) -> float:
    try:
        # If the value is already a number (int or float), return it
        if isinstance(val, (int, float)):
            return float(val)

        # If the value is a string, try to extract a number from it
        num = extract_first_number(str(val))  # Ensure it's a string before parsing

        # If a number was successfully extracted, return it
        if num is not None:
            return num
        else:
            return 0  # Could not extract a number from the string
    except Exception:
        return 0  # On any unexpected error, return 0


def avg_surgery_diff(surgery1, surgery2, surgery3):
    # Extract all surgery dates into a list and drop missing values
    dates = [surgery1, surgery2, surgery3]
    dates = [d for d in dates if pd.notnull(d)]

    # If less than 2 surgeries, difference can't be computed
    if len(dates) < 2:
        return 0

    # Sort dates to be chronological
    dates.sort()

    # Calculate differences between consecutive surgeries
    diffs = [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)]

    # Return average difference
    return np.mean(diffs)


def keyword_surgery_score(surgery_strings, surgery_keywords):
    """
    surgery_strings: list of surgery description strings (up to 3)
    Returns combined score reflecting all detected surgery categories.
    """
    matched_categories = set()  # To avoid double counting same category multiple times

    for surgery in surgery_strings:
        if pd.isna(surgery) or surgery.strip() == '':
            continue

        surgery_upper = surgery.upper()  # case insensitive matching

        # Check each category keywords
        for score, keywords in surgery_keywords.items():
            if any(keyword in surgery_upper for keyword in keywords):
                matched_categories.add(score)

    # Sum all unique category scores for this patient
    return sum(matched_categories)


def aggressiveness_by_activity_timing(diagnosis_date, activity_date):

    try:
        if isinstance(diagnosis_date, str):
            diagnosis_date = datetime.fromisoformat(diagnosis_date).date()  # convert and keep only date
        elif isinstance(diagnosis_date, datetime):
            diagnosis_date = diagnosis_date.date()

        if isinstance(activity_date, str):
            activity_date = datetime.fromisoformat(activity_date).date()  # convert and keep only date
        elif isinstance(activity_date, datetime):
            activity_date = activity_date.date()
    except Exception:
        return 0  # Invalid date format

    diff = (diagnosis_date - activity_date).days

    if diff <= 0:
        return 0

    if diff > 180:
        return 1
    elif diff > 90:
        return 2
    elif diff > 30:
        return 3
    else:
        return 4


def rank_surgery_by_name(surgery_name):

    if not surgery_name or pd.isna(surgery_name):
        return 0  # No valid surgery info

    name = str(surgery_name).lower()

    # Define keyword groups by severity
    major_keywords = ['mastectomy', 'כריתה', 'סרטן', 'tumor', 'גידול', 'oncology', 'removal']
    moderate_keywords = ['biopsy', 'ביופסיה', 'excision', 'הוצאה']
    minor_keywords = ['drainage', 'נקז', 'drain', 'cleaning', 'ניקוי']

    # Check for major keywords
    if any(k in name for k in major_keywords):
        return 3

    # Check for moderate keywords
    if any(k in name for k in moderate_keywords):
        return 2

    # Check for minor keywords
    if any(k in name for k in minor_keywords):
        return 1

    # Default if no keywords matched
    return 1


def preprocess(data: pd.DataFrame,
               data_complete: DataComplete = DefaultValue(),
               normalize: bool = True) -> pd.DataFrame:
    data[Columns.FORM] = data[Columns.FORM].map({
        "אומדן סימפטומים ודיווח סיעודי": 3,
        "אנמנזה סיעודית": 1,
        "אנמנזה סיעודית קצרה": 1,
        "אנמנזה רפואית": 4,
        "אנמנזה רפואית המטו-אונקולוגית": 5,
        "ביקור במרפאה": 4,
        "ביקור במרפאה המטו-אונקולוגית": 5,
        "ביקור במרפאה קרינה": 7,
        "דיווח סיעודי": 2,
        None: np.nan
    })
    data.drop(columns=Columns.HOSPITAL, inplace=True)
    data.drop(columns=Columns.USER_NAME, inplace=True)
    # validate age
    data[Columns.AGE] = data[Columns.AGE].where(
        (data[Columns.AGE] >= 1) & (data[Columns.AGE] <= 100), 0)
    data[Columns.BASIC_STAGE] = data[Columns.BASIC_STAGE].apply(replace_contains(
        [
            ("null", 0),
            ("c", 1),
            ("p", 2),
            ("r", 3),
        ],
        strong=True,
        mapper=to_lower
    ))
    data[Columns.HER2] = data[Columns.HER2].apply(filter_her2)
    data[Columns.HISTOLOGICAL_DIAGNOSIS] = data[
        Columns.HISTOLOGICAL_DIAGNOSIS].apply(filter_histological_diagnosis)
    data[Columns.HISTOLOGICAL_DEGREE] = data[
        Columns.HISTOLOGICAL_DEGREE].apply(replace_contains(
        [
            ("g1", 1),
            ("g2", 2),
            ("g3", 3),
            ("g4", 4)
        ],
        mapper=to_lower
    ))
    data.drop(columns=Columns.LYMPHOVASCULAR_INVASION, inplace=True)
    data[Columns.K167] = data[Columns.K167].apply(filter_k167)
    data[Columns.LYMPHATIC_PENETRATION] = data[
        Columns.LYMPHATIC_PENETRATION].apply(replace_contains(
        [
            ("NULL", 0),
            ("L0", 1),
            ("L1", 2),
            ("L2", 3)
        ],
        mapper=to_upper
    ))
    data[Columns.METASTASES_MARK] = data[
        Columns.METASTASES_MARK].apply(replace_contains(
            [
                ("m0", 0),
                ("mx", 1),
                ("m1a", 3),
                ("m1b", 4),
                ("m1", 2)
            ],
            mapper=to_lower,
            default=1
        ))
    data[Columns.MARGIN_TYPE] = data[Columns.MARGIN_TYPE].map({
        "נקיים": 0,
        "ללא": 1,
        "נגועים": 2,
        None: MISSING_VALUE_DEFAULT
    })
    data[Columns.LYMPH_NODES_MARK] = data[
        Columns.LYMPH_NODES_MARK].apply(filter_tnm_n)
    data[Columns.NODES_EXAM] = data[
        Columns.NODES_EXAM].apply(filter_nodes_exam)
    data[Columns.POSITIVE_NODES] = data[
        Columns.POSITIVE_NODES].apply(filter_positive_nodes)
    data[Columns.SIDE] = data[Columns.SIDE].map({
        "": np.nan,
        None: np.nan,
        "שמאל": 1,
        "ימין": 1,
        "דו צדדי": 2
    })
    data[Columns.STAGE] = data[Columns.STAGE].apply(replace_contains(
        [
            ("stage0is", 0.0),
            ("stage0a", 0.1),
            ("stage0", 0.2),
            ("stage1a", 1.1),
            ("stage1b", 1.2),
            ("stage1c", 1.3),
            ("stage1", 1.0),
            ("stage2a", 2.1),
            ("stage2b", 2.2),
            ("stage2", 2.0),
            ("stage3a", 3.1),
            ("stage3b", 3.2),
            ("stage3c", 3.3),
            ("stage3", 3.0),
            ("stage4", 4.0),
            ("la", 3.5),  # Or use None
        ],
        mapper=to_lower
    ))
    for col in Columns.SURGERY_DATES_NAMES:
        data[col] = pd.to_datetime(data[col], errors='coerce')
    data[Columns.SURGERY_DATE_AVERAGE_WAIT] = data.apply(
        lambda row: avg_surgery_diff(
            row[Columns.SURGERY_DATE1],
            row[Columns.SURGERY_DATE2],
            row[Columns.SURGERY_DATE3]
        ),
        axis=1
    )
    data.drop(columns=Columns.SURGERY_DATES_NAMES, inplace=True)
    surgery_keywords = {
        5: ['RADICAL MODIFIED', 'BILATERAL RADICAL', 'EXTENDED RADICAL'],
        # Radical/Extensive Mastectomy
        4: ['MASTECTOMY', 'SUBTOTAL', 'UNILATERAL SIMPLE', 'BILATERAL SIMPLE',
            'EXTENDED SIMPLE'],  # Simple/Subtotal Mastectomy
        3: ['LUMPECTOMY', 'LOCAL EXCISION', 'QUADRANTECTOMY', 'SENTINEL NODE'],
        # Lumpectomy/Local excision
        2: ['AXILLARY LYMPH NODE', 'RADICAL EXCISION', 'SIMPLE EXCISION',
            'SOLITARY LYMPH NODE'],  # Lymph node surgeries
        1: ['BIOPSY', 'EXCISION OF ECTOPIC'],  # Biopsy procedures
        0: ['HEPATECTOMY', 'LOBECTOMY', 'SALPINGO-OOPHORECTOMY'],
        # Other/unrelated
    }
    data[Columns.SURGERY_NAMES_SCORE] = data.apply(
        lambda row: keyword_surgery_score(
            [row[column] for column in Columns.SURGERY_NAMES_NAMES],
            surgery_keywords
        ),
        axis=1
    )
    data.drop(columns=Columns.SURGERY_NAMES_NAMES, inplace=True)
    data[Columns.SURGERY_SUM] = data[Columns.SURGERY_SUM].fillna(0)
    data[Columns.TUMOR_MARK] = data[Columns.TUMOR_MARK].apply(replace_contains(
        [
            ("tis", 0.5),
            ("t0", 0),
            ("t1mic", 0.8),
            ("t1a", 1),
            ("t1b", 1.3),
            ("t1c", 1.7),
            ("t1", 1.5),
            ("t2", 2),
            ("t3", 3),
            ("t4", 4)
        ],
        mapper=to_lower
    ))
    data.drop(columns=Columns.TUMOR_DEPTH, inplace=True)
    data.drop(columns=Columns.TUMOR_WIDTH, inplace=True)
    data[Columns.ER] = data[Columns.ER].apply(filter_er)
    data[Columns.PR] = data[Columns.PR].apply(filter_pr)
    data[Columns.AGGRESSIVENESS] = data.apply(
        lambda row: aggressiveness_by_activity_timing(
            row[Columns.DIAGNOSIS_DATE],
            row[Columns.SURGERY_BEFORE_AFTER_ACTIVITY_DATE]
        ),
        axis=1
    )
    data.drop(columns=Columns.SURGERY_BEFORE_AFTER_ACTIVITY_DATE, inplace=True)
    data.drop(columns=Columns.DIAGNOSIS_DATE, inplace=True)
    data[Columns.SURGERY_BEFORE_AFTER_ACTUAL_ACTIVITY] = data[
        Columns.SURGERY_BEFORE_AFTER_ACTUAL_ACTIVITY].apply(rank_surgery_by_name)
    data.drop(columns=Columns.ID, inplace=True)

    data = data_complete.complete(data)
    if normalize:
        scaled_array = StandardScaler().fit_transform(data)
        data = pd.DataFrame(scaled_array, columns=data.columns,
                            index=data.index)

    return data


if __name__ == "__main__":
    df = pd.read_csv('../train_test_splits/train_split.feats.csv',
                     encoding='utf-8-sig')
    # print(df.columns)
    print(preprocess(df).info())
    df.to_csv("temp.csv", encoding='utf-8-sig')
    # print(df.iloc[:, :].info())
