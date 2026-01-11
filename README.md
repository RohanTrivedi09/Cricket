🏏 Optimal Player Selection System for Cricket Teams

A machine learning–based system designed to assist cricket team selection by analyzing historical player performance data and reducing subjectivity in decision-making.

This project applies data preprocessing, feature engineering, and supervised machine learning to evaluate player suitability and generate insights such as player rankings, role classification, and a suggested Playing XI.

⸻

📌 Project Overview

Cricket team selection is traditionally influenced by subjective judgment, intuition, and recent impressions. This project aims to introduce a data-driven and transparent approach to player evaluation using machine learning.

The system:
	•	Analyzes historical cricket performance data (IPL-based datasets)
	•	Computes meaningful performance metrics
	•	Classifies players as Fit or Not Fit using a trained ML model
	•	Provides an interactive web interface for exploration and comparison

⸻

🎯 Objectives
	•	Analyze historical player performance data
	•	Engineer batting, bowling, and combined performance features
	•	Design an Overall Score representing a player’s total impact
	•	Train a machine learning model to assess player suitability
	•	Build an interactive Streamlit-based dashboard
	•	Generate a balanced Playing XI using rule-based heuristics

⸻

📂 Dataset

The dataset is compiled from publicly available IPL match scorecards and player statistics.

Key attributes include:
	•	Batting: total runs, balls faced, batting average, strike rate
	•	Bowling: wickets taken, economy rate, bowling average, strike rate
	•	Experience: matches played
	•	Derived features: wickets per match, overall score

Target variable:
	•	fit_label → Binary classification
	•	1: Fit player
	•	0: Not fit player

⸻

🧹 Data Preprocessing & Feature Engineering
	•	Removal of duplicates and inconsistent records
	•	Handling missing values using domain-aware defaults
	•	Standardization of player names across datasets
	•	Feature engineering:
	•	Batting Average = Runs / Dismissals
	•	Wickets per Match = Wickets / Matches
	•	Overall Score = Weighted combination of batting and bowling impact

The Overall Score is a relative metric used for ranking players, not a fixed-scale rating.

⸻

🧠 Batting & Bowling Style Classification

To add interpretability, players are categorized using rule-based logic:

Batting Styles:
	•	Explosive: Very high strike rate
	•	Aggressive: High strike rate
	•	Anchor: Balanced scoring and stability
	•	Defensive: Lower strike rate
	•	Unclassified: Insufficient match data

Bowling Styles:
	•	Very Economical
	•	Economical
	•	Average
	•	Run-leaking
	•	Unclassified: Limited bowling data

These classifications are heuristic-based and designed for explainability.

⸻

🤖 Machine Learning Model
	•	Model used: Random Forest Classifier
	•	Why Random Forest?
	•	Handles non-linear relationships
	•	Robust to noise
	•	Works well with mixed feature importance
	•	Reduces overfitting compared to single decision trees

Evaluation Metrics:
	•	Accuracy
	•	Precision
	•	Recall
	•	F1-score

The final model achieved ~94–96% accuracy on the test set.

⸻

🏏 Playing XI Generation

The system generates a suggested Playing XI using a predefined, rule-based heuristic, ensuring:
	•	Top-performing batters
	•	Specialist bowlers
	•	At least one all-rounder
	•	Overall team balance

⚠️ Note:
The Playing XI selection is deterministic and heuristic-based, intentionally kept simple for interpretability.
Advanced optimization-based selection is planned as future work.

⸻

🌐 Web Application (Streamlit)

The project is deployed locally using Streamlit and includes:
	•	Player filtering by performance and role
	•	Fit vs Not Fit classification results
	•	Overall score–based ranking
	•	Player comparison (side-by-side stats)
	•	Suggested Playing XI generation

⸻

🚀 Future Enhancements
	•	Role-constrained team selection (e.g., fixed number of anchors, finishers)
	•	Form-based and opposition-aware selection
	•	Optimization algorithms for XI generation
	•	Cloud deployment for broader accessibility
	•	Inclusion of injury and fitness data

⸻

🛠️ Tech Stack
	•	Language: Python
	•	Libraries: Pandas, NumPy, Scikit-learn, Joblib
	•	Model: Random Forest Classifier
	•	Frontend: Streamlit

⸻

📌 Disclaimer

This project is intended for academic and analytical purposes.
It does not claim to replace expert selectors but aims to assist decision-making using data-driven insights.
