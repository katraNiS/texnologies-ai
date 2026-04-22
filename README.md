# Greece Property Listings - ML Web App

Web application για ανάλυση και πρόβλεψη τιμών ακινήτων στην Ελλάδα, χρησιμοποιώντας Machine Learning.

## Dataset
Greece Property Listings - Kaggle ((https://www.kaggle.com/datasets/argyrisanastopoulos/greece-property-listings/data)

## Τεχνολογίες
- Python
- Streamlit
- Pandas, NumPy
- Scikit-learn
- Plotly

## Εγκατάσταση


## Εκτέλεση

### Online
Η εφαρμογή είναι διαθέσιμη online: https://texnologies-ai-dzcih8hdt7rugm2kl6kjgb.streamlit.app

### Τοπικά

```
pip install -r requirements.txt
streamlit run Home.py
```

## Δομή Εφαρμογής
- **Home.py** - Αρχική σελίδα
- **pages/Data_Loading.py** - Φόρτωση & Προεπεξεργασία δεδομένων
- **pages/EDA.py** - Διερευνητική Ανάλυση (Histogram, Box Plot, Scatter Plot, Heatmap, PCA)
- **pages/ML_Pipeline.py** - Regression (Linear Regression vs Random Forest) & Classification (Decision Tree vs KNN)
