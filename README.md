🚗 KSI Collision Prediction API 🚦
This project is a Flask-based API powered by a Machine Learning model to predict the likelihood of fatal traffic collisions in Toronto using the KSI (Killed or Seriously Injured) dataset. 
It includes a simple frontend interface for manual predictions and an API endpoint for automated predictions.

KSI_Project/
├── app/                    
│   ├── __init__.py            # Initializes Flask app
│   ├── main.py                # Entry point to run Flask app
│   ├── routes.py              # Flask API routes
│   ├── model/                 
│   │   └── model.pkl          # Trained machine learning model
│   ├── templates/             # HTML templates for the frontend
│   │   └── index.html         # Main prediction page
│   ├── static/                # Static files (CSS, JS, images)
│   │   └── css/styles.css     # Styles for the frontend
├── data/                      
│   └── PASSENGER_KSI.csv      # Original dataset
├── scripts/                   
│   └── train_model.py         # Data preprocessing, model training, and saving
├── requirements.txt           # Python package dependencies
├── README.md                  # Project documentation
└── report.pdf                 # Final project report
