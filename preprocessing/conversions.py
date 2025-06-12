class Columns:
    FORM = "Form Name"
    HOSPITAL = "Hospital"
    USER_NAME = "User Name"
    AGE = "אבחנה-Age"
    BASIC_STAGE = "אבחנה-Basic stage"
    DIAGNOSIS_DATE = "אבחנה-Diagnosis date"
    HER2 = "אבחנה-Her2"
    HISTOLOGICAL_DIAGNOSIS = "אבחנה-Histological diagnosis"
    HISTOLOGICAL_DEGREE = "אבחנה-Histopatological degree"
    LYMPHOVASCULAR_INVASION = 'אבחנה-Ivi -Lymphovascular invasion'
    K167 = "אבחנה-KI67 protein"
    LYMPHATIC_PENETRATION = "אבחנה-Lymphatic penetration"
    METASTASES_MARK = "אבחנה-M -metastases mark (TNM)"
    MARGIN_TYPE = "אבחנה-Margin Type"
    LYMPH_NODES_MARK = "אבחנה-N -lymph nodes mark (TNM)"
    NODES_EXAM = "אבחנה-Nodes exam"
    POSITIVE_NODES = "אבחנה-Positive nodes"
    SIDE = "אבחנה-Side"
    STAGE = "אבחנה-Stage"
    SURGERY_DATE1 = "אבחנה-Surgery date1"
    SURGERY_DATE2 = "אבחנה-Surgery date2"
    SURGERY_DATE3 = "אבחנה-Surgery date3"
    SURGERY_NAME1 = "אבחנה-Surgery name1"
    SURGERY_NAME2 = "אבחנה-Surgery name2"
    SURGERY_NAME3 = "אבחנה-Surgery name3"
    SURGERY_SUM = "אבחנה-Surgery sum"
    TUMOR_MARK = "אבחנה-T -Tumor mark (TNM)"
    TUMOR_DEPTH = "אבחנה-Tumor depth"
    TUMOR_WIDTH = "אבחנה-Tumor width"
    ER = "אבחנה-er"
    PR = "אבחנה-pr"
    SURGERY_BEFORE_AFTER_ACTIVITY_DATE = "surgery before or after-Activity date"
    SURGERY_BEFORE_AFTER_ACTUAL_ACTIVITY = "surgery before or after-Actual activity"
    ID = "id-hushed_internalpatientid"


month_prefix = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec"
]


def equiv_high(val: str) -> bool:
    # assumes val is lower case
    return "high" in val


def equiv_mid(val: str) -> bool:
    return "mid" in val or "intermediate" in val


def equiv_low(val: str) -> bool:
    return "low" in val


def equiv_pos(val: str) -> bool:
    return "pos" in val or "jhuch" in val


def equiv_neg(val: str) -> bool:
    return "neg" in val or "akhkh" in val
