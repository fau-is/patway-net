# ToDo
- Bpi2011 aus dem code nehmen
- Beim MIMIC unten noch erklären, wie wir die 3 Diagnose features konstruiert haben.


# Explainable Predictions for Medical Pathways (XPreM)


# Creation of MIMIC event log from MIMIC III database
- join tables "admission" and "patients" via subject id
- create activities and time stamps by admission location and discharge location and the according timestamps
- remove unneccassry and redundant features
- transform incorrect timestamps to get consistent formatting
- calculate age and age of death
- for patients with age >89 no real value was given (see description of MIMIC). therefore, we set the age to 95 as an assumption of the average age over 89
- calculate the age of death for patients over 89 based on the assigned age of 95 and the difference between the admission and death timestamp
- add missing information for context features




