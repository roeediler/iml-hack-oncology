from datetime import datetime

def calculate_days_diff(date_str1, date_str2):
    try:
        # Convert strings to datetime objects (ignore time)
        date1 = datetime.strptime(date_str1.split()[0], '%d/%m/%Y')
        date2 = datetime.strptime(date_str2.split()[0], '%d/%m/%Y')
        diff = (date2 - date1).days
        return diff if diff >= 0 else 0  # במידה והתוצאה שלילית, תחזיר 0
    except Exception:
        return 0

import pandas as pd

def count_surgeries(date1, date2, date3):
    dates = [date1, date2, date3]
    count = 0
    for d in dates:
        try:
            # Check if date is missing or empty string
            if pd.isna(d) or str(d).strip() == "":
                continue  # skip missing or empty dates
            # Try to convert the value to a datetime object to verify it's a valid date
            pd.to_datetime(d)
            count += 1  # valid date found, increment count
        except Exception:
            # If conversion fails, it's an invalid date, so ignore it
            continue
    return count

# Usage example with a DataFrame `df`:
df['number_of_surgeries'] = df.apply(
    lambda row: count_surgeries(row['Surgery date1'], row['Surgery date2'], row['Surgery date3']),
    axis=1  # apply function row-wise
)

def filter_nodes_exam(val) -> float:
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


from datetime import datetime


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



