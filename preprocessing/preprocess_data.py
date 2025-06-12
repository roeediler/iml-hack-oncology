import pandas as pd
import re
from typing import Callable, Optional, Any

from conversions import (Columns, month_prefix, equiv_low, equiv_mid,
                         equiv_high, equiv_neg, equiv_pos)

MISSING_VALUE_DEFAULT = -1

# Read the original CSV file

#
# df.to_csv("train.feat2.csv", index=False, encoding='utf-8-sig')
# # Create a dictionary to store the unique values of each column
# unique_values_dict = {}
#
# # Iterate over columns and store unique values
# for col in df.columns:
#     unique_values_dict[col] = df[col].unique()
#
# # Convert the dictionary to a DataFrame
# unique_values_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in unique_values_dict.items()]))
#
# # Save the unique values DataFrame to a new CSV file
# unique_values_df.to_csv('unique_values.csv', index=False, encoding='utf-8-sig')

# for col in df.columns:
#     print(f"Column '{col}' unique values:")
#     print(df[col].unique())
#     print()

# Extract the first 20 rows
# df_subset = df.head(20)

# print(df.columns)

# Save to a new CSV file (Excel-compatible)
# df_subset.to_csv('output.csv', index=False, encoding='utf-8-sig')


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
        if mapper is not None:
            val = mapper(val)
        if strong:
            val = val.split(" -,")
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


# def filter_lymphatic_penetration


def preprocess(data: pd.DataFrame) -> pd.DataFrame:
    data[Columns.BASIC_STAGE].apply(replace_contains(
        [
            ("null", 0),
            ("c", 1),
            ("p", 2),
            ("r", 3),
        ],
        strong=True,
        mapper=to_lower
    ))
    data[Columns.HER2].apply(filter_her2)
    data[Columns.HISTOLOGICAL_DIAGNOSIS].apply(filter_histological_diagnosis)
    data[Columns.HISTOLOGICAL_DEGREE].apply(replace_contains(
        [
            ("g1", 1),
            ("g2", 2),
            ("g3", 3),
            ("g4", 4)
        ],
        mapper=to_lower
    ))

    data.drop(columns=Columns.LYMPHOVASCULAR_INVASION)
    data[Columns.K167].apply(filter_k167)
    data[Columns.LYMPHATIC_PENETRATION].apply(replace_contains(
        [
            ("NULL", 0),
            ("L0", 1),
            ("L1", 2),
            ("L2", 3)
        ],
        mapper=to_upper,
        default=0
    ))

    data[Columns.SIDE].map({
        "": 0,
        None: 0,
        "שמאל": 1,
        "ימין": 1,
        "דו צדדי": 2
    })
    data[Columns.METASTASES_MARK].apply(replace_contains(
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
    data[Columns.MARGIN_TYPE].map({
        "נקיים": 0,
        "ללא": 1,
        "נגועים": 2,
        None: 3
    })

    return data


if __name__ == "__main__":
    df = pd.read_csv('../train_test_splits/train.feats.csv',
                     encoding='utf-8-sig')
    print(df.columns)
    preprocess(df)
