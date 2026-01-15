# Fraud Detection Pipeline

End-to-end machine learning system for detecting fraudulent transactions using a **production-ready data pipeline**, **data quality contracts**, and **modular ML architecture**.

This project simulates how fraud detection systems are built in real companies, from raw data ingestion to model-ready datasets and future real-time deployment.

---

## Project Objective

Build a **robust, scalable, and auditable fraud detection pipeline** that:

- Ensures **data quality from ingestion**
- Applies **clear separation of pipeline responsibilities**
- Supports **future real-time inference**
- Is understandable by both **technical and administrative stakeholders**

---

##  Architecture Overview

```text
Raw Data (CSV / Kaggle)
        ↓
Data Ingestion & Validation (Pandera)
        ↓
Exploratory Data Analysis (EDA)
        ↓
Feature Engineering
        ↓
Model Training (next phase)
        ↓
Evaluation & Monitoring (planned)
        ↓
API / Streaming (planned)


