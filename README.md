# FACETS-LLM
Forecasting and AI-Driven Customer Segmentation System (a.k.a. FACETS). Meant to enhance business decision-making by providing  actionable insights in natural language but also bridge the gap between complex machine learning models and non-technical business users. 

## Project Structure
```
FACETS_Project/
│
├── docs/
│   ├── project_proposal.pdf
│   ├── literature_review.md
│   ├── final_report.docx
│   └── presentation.pptx
│
├── data/
│   ├── raw/
│   │   ├── rossman_sales.csv
│   │   └── online_retail_II.csv
│   ├── processed/
│   │   ├── sales_processed.csv
│   │   └── customers_processed.csv
│   └── interim/
│       └── (temporary processed files)
│
├── notebooks/
│   ├── exploratory_data_analysis.ipynb
│   ├── data_preprocessing.ipynb
│   ├── demand_forecasting_baseline.ipynb
│   ├── demand_forecasting_advanced.ipynb
│   ├── customer_segmentation.ipynb
│   ├── llm_insight_generation.ipynb
│   └── visualization_and_reporting.ipynb
│
├── scripts/
│   ├── preprocessing.py
│   ├── forecasting_models.py
│   ├── clustering_models.py
│   └── llm_insights.py
│
├── models/
│   ├── forecasting/
│   │   ├── arima_model.pkl
│   │   ├── lstm_model.pth
│   │   └── transformer_model.pt
│   └── clustering/
│       ├── kmeans_model.pkl
│       └── dbscan_model.pkl
│
├── outputs/
│   ├── visualizations/
│   │   └── (graphs, charts, interactive visualizations)
│   └── llm_reports/
│       └── (LLM-generated insights and summaries)
│
├── evaluation/
│   ├── forecasting_evaluation.md
│   ├── clustering_evaluation.md
│   └── llm_evaluation.md
│
├── app/ (optional if building interactive dashboards)
│   ├── frontend/
│   └── backend/
│
├── requirements.txt
├── environment.yml
├── README.md
└── .gitignore

```