# PCOS Clinical Decision Support

A web application for retrieving a patient record and reviewing a
structured PCOS risk assessment: estimated probability, risk tier,
contributing factors, Rotterdam criteria, comorbidity considerations, and
recommended clinical actions.

## Running the application

```bash
cd pcos_clinic_app
python3 -m pip install -r requirements.txt
python3 app.py
```

The application is served at `http://127.0.0.1:8000`. Enter a patient "NRIC" (CSV row number for this protoype)
to retrieve the assessment.

## Structure

```
pcos_clinic_app/
├── app.py              Web application (Flask)
├── requirements.txt
├── templates/          HTML templates
├── static/             Stylesheet
├── src/                Assessment pipeline modules
├── models/             Trained classifier
└── data/               Patient records and model artifacts
```

## Notes

- The risk model was trained on a single-site clinical dataset and has not
  been externally validated.
- The tool supports clinical assessment and does not replace clinical
  judgment or constitute a diagnosis.
- Future iterations of this product should involve refinement of assessments from a medically trained clinician or doctor.
