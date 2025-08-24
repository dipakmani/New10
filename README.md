"""
generate_25_healthcare_tables.py

- Generates 25 interrelated healthcare tables (CSV output).
- Ensures Country -> State -> City consistency across records for entities that include location.
- PARAMETER: ROWS_PER_TABLE controls how many rows per table (default 50k for safe test).
- WARNING: Setting ROWS_PER_TABLE = 500000 will create large files and require lots of RAM/disk.

Dependencies:
 pip install pandas numpy faker
"""

import os
import numpy as np
import pandas as pd
from faker import Faker
from datetime import datetime, timedelta

fake = Faker()
Faker.seed(42)
np.random.seed(42)

# ---------- CONFIG ----------
OUT_DIR = "healthcare_25tables_output"
os.makedirs(OUT_DIR, exist_ok=True)

# Default rows per table (change to 500_000 for 5 lakh once tested)
ROWS_PER_TABLE = 50_000

# Pools sizes for dimension-like entities (smaller than ROWS_PER_TABLE)
NUM_PATIENTS = ROWS_PER_TABLE
NUM_DOCTORS = max(2000, ROWS_PER_TABLE // 20)
NUM_DEPARTMENTS = max(50, ROWS_PER_TABLE // 1000)
NUM_WARDS = max(100, ROWS_PER_TABLE // 5000)
NUM_BEDS = max(2000, ROWS_PER_TABLE // 25)
NUM_MEDICATIONS = 2000
NUM_EQUIP = max(500, ROWS_PER_TABLE // 100)

# 5 countries and matching states & cities (kept small for example; add more if you want)
COUNTRIES = ["India", "USA", "UK", "Australia", "Canada"]
STATES_BY_COUNTRY = {
    "India": ["MH", "DL", "KA", "TN", "GJ"],
    "USA": ["CA", "NY", "TX", "FL", "IL"],
    "UK": ["England", "Scotland", "Wales", "N. Ireland", "London"],
    "Australia": ["NSW", "VIC", "QLD", "WA", "SA"],
    "Canada": ["ON", "QC", "BC", "AB", "MB"]
}
CITIES_BY_STATE = {
    # India
    "MH": ["Mumbai","Pune","Nashik"],
    "DL": ["New Delhi","Noida","Gurgaon"],
    "KA": ["Bengaluru","Mysore","Mangalore"],
    "TN": ["Chennai","Coimbatore","Madurai"],
    "GJ": ["Ahmedabad","Surat","Vadodara"],
    # USA
    "CA": ["Los Angeles","San Francisco","San Diego"],
    "NY": ["New York City","Buffalo","Albany"],
    "TX": ["Houston","Dallas","Austin"],
    "FL": ["Miami","Orlando","Tampa"],
    "IL": ["Chicago","Springfield","Aurora"],
    # UK
    "England": ["London","Manchester","Birmingham"],
    "Scotland": ["Edinburgh","Glasgow","Aberdeen"],
    "Wales": ["Cardiff","Swansea","Newport"],
    "N. Ireland": ["Belfast","Derry","Lisburn"],
    "London": ["City of London","Westminster","Camden"],
    # Australia
    "NSW": ["Sydney","Newcastle","Wollongong"],
    "VIC": ["Melbourne","Geelong","Ballarat"],
    "QLD": ["Brisbane","Gold Coast","Cairns"],
    "WA": ["Perth","Fremantle","Bunbury"],
    "SA": ["Adelaide","Mount Gambier","Gawler"],
    # Canada
    "ON": ["Toronto","Ottawa","Hamilton"],
    "QC": ["Montreal","Quebec City","Laval"],
    "BC": ["Vancouver","Victoria","Richmond"],
    "AB": ["Calgary","Edmonton","Red Deer"],
    "MB": ["Winnipeg","Brandon","Steinbach"],
}

# Small helper functions
def pick_state_country(n):
    countries = np.random.choice(COUNTRIES, size=n)
    states = [np.random.choice(STATES_BY_COUNTRY[c]) for c in countries]
    cities = [np.random.choice(CITIES_BY_STATE[s]) for s in states]
    return countries, np.array(states), np.array(cities)

def ids(prefix, n, width=6):
    return np.array([f"{prefix}{i:0{width}d}" for i in range(1, n+1)])

# Create ID pools (these will be referenced across tables)
PATIENT_IDS = ids("PAT", NUM_PATIENTS, width=7)
DOCTOR_IDS  = ids("DOC", NUM_DOCTORS, width=6)
DEPARTMENT_IDS = ids("DEP", NUM_DEPARTMENTS, width=4)
WARD_IDS = ids("WARD", NUM_WARDS, width=4)
BED_IDS = ids("BED", NUM_BEDS, width=6)
MED_IDS = ids("MED", NUM_MEDICATIONS, width=5)
EQUIP_IDS = ids("EQ", NUM_EQUIP, width=5)
VISIT_IDS = ids("VIS", ROWS_PER_TABLE, width=8)   # primary fact ID for visit-like tables
APPT_IDS = ids("APT", ROWS_PER_TABLE, width=8)
BILL_IDS = ids("BIL", ROWS_PER_TABLE, width=8)
PRESC_IDS = ids("PRS", ROWS_PER_TABLE, width=8)
LAB_IDS = ids("LAB", ROWS_PER_TABLE, width=8)
RAD_IDS = ids("RAD", ROWS_PER_TABLE, width=8)
SURG_IDS = ids("SUR", ROWS_PER_TABLE, width=8)
CLAIM_IDS = ids("CLM", ROWS_PER_TABLE, width=8)
PAY_IDS = ids("PAY", ROWS_PER_TABLE, width=8)
FEED_IDS = ids("FDB", ROWS_PER_TABLE, width=8)
EMG_IDS = ids("EMG", ROWS_PER_TABLE, width=8)
BEDALLOC_IDS = ids("BA", ROWS_PER_TABLE, width=8)
PROC_IDS = ids("PRC", ROWS_PER_TABLE, width=8)
STAFF_IDS = ids("STF", max(NUM_DOCTORS, ROWS_PER_TABLE//10), width=6)
SHIFT_IDS = ids("SH", 10, width=3)
INV_IDS = ids("INV", max(1000, ROWS_PER_TABLE//50), width=6)

# Base timestamp generators
start_date = datetime(2023,1,1)
def random_dates(n, start=start_date, days_span=600):
    offs = np.random.randint(0, days_span, size=n)
    return np.array([ (start + timedelta(days=int(d))).strftime("%Y-%m-%d") for d in offs ])

# ---------- TABLE GENERATORS ----------
# Each table will have 12 attributes (including its ID). We'll make sure location columns are consistent.

# 1) Patients
def make_patients(n):
    countries, states, cities = pick_state_country(n)
    df = pd.DataFrame({
        "PatientID": ids("PAT", n, width=7),
        "FullName": [fake.name() for _ in range(n)],
        "Gender": np.random.choice(["Male","Female"], size=n),
        "DOB": random_dates(n, start=datetime(1940,1,1), days_span=30000),
        "Country": countries,
        "State": states,
        "City": cities,
        "Phone": [fake.phone_number() for _ in range(n)],
        "Email": [fake.email() for _ in range(n)],
        "BloodGroup": np.random.choice(["A+","A-","B+","B-","O+","O-","AB+","AB-"], size=n),
        "PrimaryLanguage": np.random.choice(["English","Hindi","Spanish","French","Mandarin"], size=n),
        "IsInsured": np.random.choice(["Yes","No"], size=n),
    })
    return df

# 2) Doctors
def make_doctors(n):
    countries, states, cities = pick_state_country(n)
    dept_choices = np.random.choice(DEPARTMENT_IDS, size=n)
    df = pd.DataFrame({
        "DoctorID": ids("DOC", n, width=6),
        "DoctorName": [fake.name() for _ in range(n)],
        "Specialization": np.random.choice(["Cardiology","Neurology","Oncology","Orthopedics","Pediatrics","General"], size=n),
        "YearsExperience": np.random.randint(1,40,size=n),
        "Country": countries,
        "State": states,
        "City": cities,
        "DepartmentID": dept_choices,
        "Email": [fake.email() for _ in range(n)],
        "Phone": [fake.phone_number() for _ in range(n)],
        "ConsultationFee": np.round(np.random.uniform(50,500, size=n),2),
        "Status": np.random.choice(["Active","On Leave"], size=n),
    })
    return df

# 3) Departments
def make_departments(n):
    # Departments typically fewer; reuse country mapping small
    countries, states, cities = pick_state_country(n)
    df = pd.DataFrame({
        "DepartmentID": ids("DEP", n, width=4),
        "DepartmentName": [f"{fake.word().capitalize()} Dept" for _ in range(n)],
        "HeadDoctorID": np.random.choice(DOCTOR_IDS, n),
        "Country": countries,
        "State": states,
        "City": cities,
        "Floor": np.random.randint(0,10, size=n),
        "ContactNo": [fake.phone_number() for _ in range(n)],
        "NoOfBeds": np.random.randint(5,200, size=n),
        "NoOfStaff": np.random.randint(5,200, size=n),
        "IsEmergencyDept": np.random.choice(["Yes","No"], size=n),
        "EstablishedYear": np.random.randint(1980,2024,size=n),
    })
    return df

# 4) Wards
def make_wards(n):
    countries, states, cities = pick_state_country(n)
    df = pd.DataFrame({
        "WardID": ids("WARD", n, width=4),
        "WardName": [f"Ward-{i}" for i in range(1,n+1)],
        "DepartmentID": np.random.choice(DEPARTMENT_IDS, n),
        "Country": countries,
        "State": states,
        "City": cities,
        "WardType": np.random.choice(["ICU","General","Private","HDU"], n),
        "Capacity": np.random.randint(5,100, n),
        "Occupancy": np.random.randint(0,100, n),
        "HeadNurseID": np.random.choice(STAFF_IDS, n),
        "PhoneExt": np.random.randint(1000,9999,n),
        "Notes": [fake.sentence(nb_words=6) for _ in range(n)],
    })
    return df

# 5) Beds
def make_beds(n):
    df = pd.DataFrame({
        "BedID": ids("BED", n, width=6),
        "WardID": np.random.choice(WARD_IDS, n),
        "BedNumber": np.random.randint(1,500, n),
        "BedType": np.random.choice(["Normal","ICU","Recovery"], n),
        "IsOccupied": np.random.choice(["Yes","No"], n),
        "OccupiedByPatientID": np.random.choice(PATIENT_IDS, n),
        "AssignedDate": random_dates(n),
        "LastCleaned": random_dates(n, days_span=90),
        "HasOxygen": np.random.choice(["Yes","No"], n),
        "ChargePerDay": np.round(np.random.uniform(100,5000,n),2),
        "RoomNumber": np.random.randint(100,900,n),
        "Notes": [fake.sentence(nb_words=6) for _ in range(n)]
    })
    return df

# 6) Staff (includes nurses/technicians)
def make_staff(n):
    countries, states, cities = pick_state_country(n)
    df = pd.DataFrame({
        "StaffID": ids("STF", n, width=6),
        "StaffName": [fake.name() for _ in range(n)],
        "Role": np.random.choice(["Nurse","Technician","Admin","Receptionist","Pharmacist"], n),
        "DepartmentID": np.random.choice(DEPARTMENT_IDS, n),
        "Country": countries,
        "State": states,
        "City": cities,
        "ShiftID": np.random.choice(SHIFT_IDS, n),
        "Phone": [fake.phone_number() for _ in range(n)],
        "Email": [fake.email() for _ in range(n)],
        "JoinDate": random_dates(n, days_span=4000),
        "Status": np.random.choice(["Active","On Leave","Resigned"], n),
        "SupervisorID": np.random.choice(STAFF_IDS, n),
    })
    return df

# 7) Shifts
def make_shifts(n):
    df = pd.DataFrame({
        "ShiftID": ids("SH", n, width=4),
        "ShiftName": [f"Shift-{i}" for i in range(1,n+1)],
        "StartTime": [f"{np.random.randint(0,23):02d}:00" for _ in range(n)],
        "EndTime": [f"{np.random.randint(0,23):02d}:00" for _ in range(n)],
        "ResponsibleDept": np.random.choice(DEPARTMENT_IDS, n),
        "AssignedStaffID": np.random.choice(STAFF_IDS, n),
        "AssignedDoctorID": np.random.choice(DOCTOR_IDS, n),
        "ShiftType": np.random.choice(["Morning","Evening","Night","OnCall"], n),
        "IsWeekend": np.random.choice(["Yes","No"], n),
        "Notes": [fake.sentence(nb_words=6) for _ in range(n)],
        "Location": np.random.choice(["North Wing","South Wing","East Wing","West Wing"], n),
        "Status": np.random.choice(["Planned","Completed"], n),
    })
    return df

# 8) Visits (main fact)
def make_visits(n):
    countries, states, cities = pick_state_country(n)
    adm = random_dates(n)
    dis = random_dates(n)
    df = pd.DataFrame({
        "VisitID": ids("VIS", n, width=8),
        "PatientID": np.random.choice(PATIENT_IDS, n),
        "DoctorID": np.random.choice(DOCTOR_IDS, n),
        "DepartmentID": np.random.choice(DEPARTMENT_IDS, n),
        "Country": countries,
        "State": states,
        "City": cities,
        "AdmissionDate": adm,
        "DischargeDate": dis,
        "VisitType": np.random.choice(["Emergency","Inpatient","Outpatient"], n),
        "LengthOfStayDays": np.random.randint(0,30,n),
        "VisitCost": np.round(np.random.uniform(100.0,20000.0,n),2),
    })
    return df

# 9) Appointments
def make_appointments(n):
    countries, states, cities = pick_state_country(n)
    df = pd.DataFrame({
        "AppointmentID": ids("APT", n, width=8),
        "PatientID": np.random.choice(PATIENT_IDS, n),
        "DoctorID": np.random.choice(DOCTOR_IDS, n),
        "DepartmentID": np.random.choice(DEPARTMENT_IDS, n),
        "Country": countries,
        "State": states,
        "City": cities,
        "AppointmentDate": random_dates(n),
        "Status": np.random.choice(["Scheduled","Completed","Cancelled"], n),
        "Channel": np.random.choice(["Phone","App","Walk-in","Referral"], n),
        "Slot": np.random.choice(["09:00","10:00","11:00","14:00","15:00"], n),
        "Notes": [fake.sentence(nb_words=6) for _ in range(n)],
        "IsFirstVisit": np.random.choice(["Yes","No"], n)
    })
    return df

# 10) Diagnosis
def make_diagnosis(n):
    df = pd.DataFrame({
        "DiagnosisID": ids("DGN", n, width=8),
        "VisitID": np.random.choice(VISIT_IDS, n),
        "PatientID": np.random.choice(PATIENT_IDS, n),
        "DoctorID": np.random.choice(DOCTOR_IDS, n),
        "DiagnosisName": np.random.choice(["Diabetes","Hypertension","Fracture","Infection","Cancer","Asthma"], n),
        "ICD10": np.random.choice(["E11","I10","S52","A09","C34","J45"], n),
        "DiagnosisDate": random_dates(n),
        "Severity": np.random.choice(["Low","Medium","High","Critical"], n),
        "IsChronic": np.random.choice(["Yes","No"], n),
        "PrimarySymptom": np.random.choice(["Pain","Fever","Cough","Bleeding"], n),
        "DepartmentID": np.random.choice(DEPARTMENT_IDS, n),
        "Notes": [fake.sentence(nb_words=6) for _ in range(n)],
    })
    return df

# 11) Treatments
def make_treatments(n):
    df = pd.DataFrame({
        "TreatmentID": ids("TRT", n, width=8),
        "DiagnosisID": np.random.choice(ids("DGN", n, width=8), n),
        "PatientID": np.random.choice(PATIENT_IDS, n),
        "DoctorID": np.random.choice(DOCTOR_IDS, n),
        "TreatmentType": np.random.choice(["Medication","Surgery","Therapy","Observation"], n),
        "StartDate": random_dates(n),
        "EndDate": random_dates(n),
        "Outcome": np.random.choice(["Recovered","Improved","No Change","Worsened"], n),
        "Cost": np.round(np.random.uniform(50,50000,n),2),
        "FollowUpRequired": np.random.choice(["Yes","No"], n),
        "DepartmentID": np.random.choice(DEPARTMENT_IDS, n),
        "Notes": [fake.sentence(nb_words=6) for _ in range(n)]
    })
    return df

# 12) Medications
def make_medications(n):
    df = pd.DataFrame({
        "MedicationID": ids("MED", n, width=6),
        "TreatmentID": np.random.choice(ids("TRT", n, width=8), n),
        "DrugName": np.random.choice(["Paracetamol","Amoxicillin","Ibuprofen","Metformin","Aspirin"], n),
        "Dose": np.random.choice(["250mg","500mg","5mg","10mg"], n),
        "Frequency": np.random.choice(["OD","BD","TDS","PRN"], n),
        "StartDate": random_dates(n),
        "EndDate": random_dates(n),
        "PrescribedBy": np.random.choice(DOCTOR_IDS, n),
        "GivenByPharmacyID": np.random.choice(ids("PHM", n, width=6), n),
        "Cost": np.round(np.random.uniform(1,2000,n),2),
        "Route": np.random.choice(["Oral","IV","IM"], n),
        "Notes": [fake.sentence(nb_words=6) for _ in range(n)]
    })
    return df

# 13) Prescriptions
def make_prescriptions(n):
    df = pd.DataFrame({
        "PrescriptionID": ids("PRS", n, width=8),
        "VisitID": np.random.choice(VISIT_IDS, n),
        "MedicationID": np.random.choice(MED_IDS, n),
        "PatientID": np.random.choice(PATIENT_IDS, n),
        "DoctorID": np.random.choice(DOCTOR_IDS, n),
        "IssuedDate": random_dates(n),
        "DurationDays": np.random.randint(1,60,n),
        "Dosage": np.random.choice(["1 tab","2 tab","5 ml"], n),
        "Refills": np.random.randint(0,3,n),
        "PharmacyName": np.random.choice(["InHouse","PharmaX","PharmaY"], n),
        "Status": np.random.choice(["Active","Completed","Cancelled"], n),
        "Notes": [fake.sentence(nb_words=5) for _ in range(n)]
    })
    return df

# 14) LabTests
def make_labtests(n):
    df = pd.DataFrame({
        "LabTestID": ids("LAB", n, width=8),
        "VisitID": np.random.choice(VISIT_IDS, n),
        "TestName": np.random.choice(["CBC","LFT","KFT","Blood Sugar","Lipid Panel"], n),
        "TestDate": random_dates(n),
        "Result": np.random.choice(["Normal","Abnormal","Critical"], n),
        "Units": np.random.choice(["mg/dL","g/dL","IU/L","%"], n),
        "ReferenceRange": np.random.choice(["Low","Normal","High"], n),
        "Cost": np.round(np.random.uniform(50,5000,n),2),
        "PerformedBy": np.random.choice(STAFF_IDS, n),
        "LabName": np.random.choice(["Central Lab","Lab-A","Lab-B"], n),
        "Status": np.random.choice(["Pending","Completed"], n),
        "Notes": [fake.sentence(nb_words=5) for _ in range(n)],
    })
    return df

# 15) Radiology
def make_radiology(n):
    df = pd.DataFrame({
        "RadiologyID": ids("RAD", n, width=8),
        "VisitID": np.random.choice(VISIT_IDS, n),
        "ScanType": np.random.choice(["X-Ray","CT","MRI","Ultrasound"], n),
        "ScanDate": random_dates(n),
        "ResultSummary": np.random.choice(["Normal","Findings","Urgent"], n),
        "ImageRef": [fake.uuid4() for _ in range(n)],
        "RadiologistID": np.random.choice(DOCTOR_IDS, n),
        "TechnicianID": np.random.choice(STAFF_IDS, n),
        "Cost": np.round(np.random.uniform(100,15000,n),2),
        "DepartmentID": np.random.choice(DEPARTMENT_IDS, n),
        "ReportAvailable": np.random.choice(["Yes","No"], n),
        "Notes": [fake.sentence(nb_words=5) for _ in range(n)]
    })
    return df

# 16) Surgeries / Procedures
def make_procedures(n):
    df = pd.DataFrame({
        "ProcedureID": ids("PRC", n, width=8),
        "VisitID": np.random.choice(VISIT_IDS, n),
        "ProcedureName": np.random.choice(["Appendectomy","Bypass","Hip Replacement","Endoscopy"], n),
        "ProcedureDate": random_dates(n),
        "SurgeonID": np.random.choice(DOCTOR_IDS, n),
        "AnesthetistID": np.random.choice(DOCTOR_IDS, n),
        "DurationMins": np.random.randint(15,600,n),
        "Outcome": np.random.choice(["Successful","Complication","Failed"], n),
        "ProcedureCost": np.round(np.random.uniform(1000,200000,n),2),
        "TheatreRoom": np.random.randint(1,50,n),
        "DepartmentID": np.random.choice(DEPARTMENT_IDS, n),
        "Notes": [fake.sentence(nb_words=6) for _ in range(n)]
    })
    return df

# 17) Billing
def make_billing(n):
    bill_date = random_dates(n)
    df = pd.DataFrame({
        "BillID": ids("BIL", n, width=8),
        "VisitID": np.random.choice(VISIT_IDS, n),
        "PatientID": np.random.choice(PATIENT_IDS, n),
        "TotalAmount": np.round(np.random.uniform(100.0,50000.0,n),2),
        "Discount": np.round(np.random.uniform(0,5000,n),2),
        "Tax": np.round(np.random.uniform(0,5000,n),2),
        "NetPayable": lambda x: x["TotalAmount"] - x["Discount"] + x["Tax"],
        "PaymentMode": np.random.choice(["Cash","Card","Insurance","UPI"], n),
        "PaymentStatus": np.random.choice(["Paid","Pending","Partially Paid"], n),
        "BillDate": bill_date,
        "DueDate": random_dates(n),
        "InsuranceID": np.random.choice(ids("INS", n, width=6), n)
    })
    # compute NetPayable properly
    df["NetPayable"] = df["TotalAmount"] - df["Discount"] + df["Tax"]
    return df

# 18) Insurance
def make_insurance(n):
    df = pd.DataFrame({
        "InsuranceID": ids("INS", n, width=6),
        "ProviderName": np.random.choice(["HealthCo","MediCare","InsureX","ProtectLife"], n),
        "PolicyNumber": [fake.bothify(text='POL-####-#####') for _ in range(n)],
        "CoveragePercent": np.random.randint(30,100,n),
        "SumInsured": np.round(np.random.uniform(10000,2000000,n),2),
        "StartDate": random_dates(n),
        "EndDate": random_dates(n),
        "PatientID": np.random.choice(PATIENT_IDS, n),
        "Active": np.random.choice(["Yes","No"], n),
        "TPAName": np.random.choice(["TPA1","TPA2","TPA3"], n),
        "ContactNumber": [fake.phone_number() for _ in range(n)],
        "Notes": [fake.sentence(nb_words=5) for _ in range(n)]
    })
    return df

# 19) Claims
def make_claims(n):
    df = pd.DataFrame({
        "ClaimID": ids("CLM", n, width=8),
        "InsuranceID": np.random.choice(ids("INS", n, width=6), n),
        "BillID": np.random.choice(BILL_IDS, n),
        "ClaimAmount": np.round(np.random.uniform(100.0,30000.0,n),2),
        "ApprovedAmount": np.round(np.random.uniform(0.0,30000.0,n),2),
        "ClaimStatus": np.random.choice(["Submitted","Approved","Rejected","Queried"], n),
        "SubmittedOn": random_dates(n),
        "ProcessedOn": random_dates(n),
        "DenialCode": np.random.choice(["NA","DOCS","ELIG","EXCL"], n),
        "AppealID": np.random.choice(ids("APL", n, width=6), n),
        "IsFinal": np.random.choice(["Yes","No"], n),
        "Notes": [fake.sentence(nb_words=4) for _ in range(n)]
    })
    return df

# 20) Payments
def make_payments(n):
    df = pd.DataFrame({
        "PaymentID": ids("PAY", n, width=8),
        "BillID": np.random.choice(BILL_IDS, n),
        "AmountPaid": np.round(np.random.uniform(10.0,50000.0,n),2),
        "PaymentMode": np.random.choice(["Cash","Card","UPI","NetBanking","Insurance"], n),
        "PaymentDate": random_dates(n),
        "TransactionRef": [fake.bothify(text='TXN-########') for _ in range(n)],
        "PatientID": np.random.choice(PATIENT_IDS, n),
        "VisitID": np.random.choice(VISIT_IDS, n),
        "ProcessedByStaffID": np.random.choice(STAFF_IDS, n),
        "DepartmentID": np.random.choice(DEPARTMENT_IDS, n),
        "Status": np.random.choice(["Completed","Pending","Failed"], n),
        "Notes": [fake.sentence(nb_words=4) for _ in range(n)]
    })
    return df

# 21) Equipment
def make_equipment(n):
    countries, states, cities = pick_state_country(n)
    df = pd.DataFrame({
        "EquipmentID": ids("EQ", n, width=6),
        "EquipmentName": [fake.word().capitalize() for _ in range(n)],
        "DepartmentID": np.random.choice(DEPARTMENT_IDS, n),
        "WardID": np.random.choice(WARD_IDS, n),
        "Country": countries,
        "State": states,
        "City": cities,
        "PurchaseDate": random_dates(n, days_span=4000),
        "Cost": np.round(np.random.uniform(500.0,500000.0,n),2),
        "Status": np.random.choice(["Operational","Maintenance","OutOfService"], n),
        "Vendor": [fake.company() for _ in range(n)],
        "LastServiced": random_dates(n, days_span=365)
    })
    return df

# 22) Inventory
def make_inventory(n):
    df = pd.DataFrame({
        "InventoryID": ids("INV", n, width=6),
        "ItemName": [fake.word().capitalize() for _ in range(n)],
        "Category": np.random.choice(["Pharmacy","Consumable","Device"], n),
        "Quantity": np.random.randint(0,5000,n),
        "ReorderLevel": np.random.randint(1,500,n),
        "UnitPrice": np.round(np.random.uniform(1.0,5000.0,n),2),
        "DepartmentID": np.random.choice(DEPARTMENT_IDS, n),
        "WardID": np.random.choice(WARD_IDS, n),
        "Supplier": [fake.company() for _ in range(n)],
        "PurchaseDate": random_dates(n, days_span=1500),
        "ExpiryDate": random_dates(n, days_span=2000),
        "Notes": [fake.sentence(nb_words=4) for _ in range(n)]
    })
    return df

# 23) Feedback
def make_feedback(n):
    df = pd.DataFrame({
        "FeedbackID": ids("FDB", n, width=8),
        "PatientID": np.random.choice(PATIENT_IDS, n),
        "VisitID": np.random.choice(VISIT_IDS, n),
        "DoctorID": np.random.choice(DOCTOR_IDS, n),
        "DepartmentID": np.random.choice(DEPARTMENT_IDS, n),
        "Rating": np.random.randint(1,6,n),
        "Category": np.random.choice(["Care","Cleanliness","WaitingTime","Billing","Facilities"], n),
        "Comment": [fake.sentence(nb_words=8) for _ in range(n)],
        "SubmittedOn": random_dates(n, days_span=365),
        "Resolved": np.random.choice(["Yes","No"], n),
        "ResponderStaffID": np.random.choice(STAFF_IDS, n),
        "ResponseTimeDays": np.random.randint(0,30,n),
    })
    return df

# 24) Emergencies
def make_emergencies(n):
    countries, states, cities = pick_state_country(n)
    df = pd.DataFrame({
        "EmergencyID": ids("EMG", n, width=8),
        "PatientID": np.random.choice(PATIENT_IDS, n),
        "VisitID": np.random.choice(VISIT_IDS, n),
        "Country": countries,
        "State": states,
        "City": cities,
        "ArrivalMode": np.random.choice(["Ambulance","WalkIn","Referral"], n),
        "TriageLevel": np.random.choice(["P1","P2","P3"], n),
        "ArrivalTime": random_dates(n, days_span=400),
        "TreatmentStart": random_dates(n, days_span=400),
        "Outcome": np.random.choice(["Admitted","Discharged","Expired"], n),
        "Notes": [fake.sentence(nb_words=4) for _ in range(n)]
    })
    return df

# 25) BedAllocations (final table)
def make_bed_allocations(n):
    df = pd.DataFrame({
        "AllocationID": ids("BA", n, width=8),
        "BedID": np.random.choice(BED_IDS, n),
        "PatientID": np.random.choice(PATIENT_IDS, n),
        "VisitID": np.random.choice(VISIT_IDS, n),
        "WardID": np.random.choice(WARD_IDS, n),
        "FromDate": random_dates(n, days_span=400),
        "ToDate": random_dates(n, days_span=400),
        "AllocatedByStaffID": np.random.choice(STAFF_IDS, n),
        "Reason": np.random.choice(["Surgery","ICU","Observation","Isolation"], n),
        "IsTransfer": np.random.choice(["Yes","No"], n),
        "CostPerDay": np.round(np.random.uniform(100,5000,n),2),
        "Notes": [fake.sentence(nb_words=4) for _ in range(n)]
    })
    return df

# ---------- DRIVER ----------
TABLE_FUNCS = [
    ("Patients.csv", make_patients, NUM_PATIENTS),
    ("Doctors.csv", make_doctors, NUM_DOCTORS),
    ("Departments.csv", make_departments, NUM_DEPARTMENTS),
    ("Wards.csv", make_wards, NUM_WARDS),
    ("Beds.csv", make_beds, NUM_BEDS),
    ("Staff.csv", make_staff, max(NUM_DOCTORS, ROWS_PER_TABLE//10)),
    ("Shifts.csv", make_shifts, max(1000, ROWS_PER_TABLE//50)),
    ("Visits.csv", make_visits, ROWS_PER_TABLE),
    ("Appointments.csv", make_appointments, ROWS_PER_TABLE),
    ("Diagnosis.csv", make_diagnosis, ROWS_PER_TABLE),
    ("Treatments.csv", make_treatments, ROWS_PER_TABLE),
    ("Medications.csv", make_medications, ROWS_PER_TABLE),
    ("Prescriptions.csv", make_prescriptions, ROWS_PER_TABLE),
    ("LabTests.csv", make_labtests, ROWS_PER_TABLE),
    ("Radiology.csv", make_radiology, ROWS_PER_TABLE),
    ("Procedures.csv", make_procedures, ROWS_PER_TABLE),
    ("Billing.csv", make_billing, ROWS_PER_TABLE),
    ("Insurance.csv", make_insurance, ROWS_PER_TABLE),
    ("Claims.csv", make_claims, ROWS_PER_TABLE),
    ("Payments.csv", make_payments, ROWS_PER_TABLE),
    ("Equipment.csv", make_equipment, NUM_EQUIP),
    ("Inventory.csv", make_inventory, max(1000, ROWS_PER_TABLE//50)),
    ("Feedback.csv", make_feedback, ROWS_PER_TABLE//2),
    ("Emergencies.csv", make_emergencies, ROWS_PER_TABLE//4),
    ("BedAllocations.csv", make_bed_allocations, ROWS_PER_TABLE//2),
]

def generate_all(out_dir=OUT_DIR, tables = TABLE_FUNCS):
    print(f"Generating {len(tables)} tables into folder: {out_dir}")
    for name, fn, n in tables:
        print(f" -> Generating {name} with {n} rows ...")
        df = fn(n)
        # Convert datetime-like to string (already strings here)
        path = os.path.join(out_dir, name)
        df.to_csv(path, index=False)
        print(f"    saved: {path} ({df.shape[0]} rows, {df.shape[1]} cols)")

if __name__ == "__main__":
    # Quick sanity: run with smaller numbers if you don't have heavy resources
    print("CONFIG:")
    print(f"ROWS_PER_TABLE (fact-like) = {ROWS_PER_TABLE}")
    print(f"NUM_PATIENTS = {NUM_PATIENTS}, NUM_DOCTORS = {NUM_DOCTORS}, NUM_DEPARTMENTS = {NUM_DEPARTMENTS}")
    generate_all()
    print("DONE.")
