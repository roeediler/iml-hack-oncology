def filter_nodes_exam(val) -> float:
    try:
        # טיפוס מספרי תקני
        if isinstance(val, (int, float)):
            return float(val)

        # נסה לחלץ מספר מהמחרוזת
        num = extract_first_number(val)
        return num if num is not None else -1 # ➜ TODO + -1

    except Exception:
        return -1 # ➜ TODO + -1


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
        return -1 # ➜ TODO + -1

    return -1 # ➜ TODO + -1



dict_form_name = {
    "אומדן סימפטומים ודיווח סיעודי": 3,
    "אנמנזה סיעודית": 1,
    "אנמנזה סיעודית קצרה": 1,
    "אנמנזה רפואית": 4,
    "אנמנזה רפואית המטו-אונקולוגית": 5,
    "ביקור במרפאה": 4,
    "ביקור במרפאה המטו-אונקולוגית": 5,
    "ביקור במרפאה קרינה": 7,
    "דיווח סיעודי": 2
}


