---
title: FACETS-LLM
emoji: 📊
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "5.25.2"
app_file: app.py
pinned: false
---

# FACETS-LLM
Forecasting and AI-Driven Customer Segmentation System (a.k.a. FACETS). Meant to enhance business decision-making by providing  actionable insights in natural language but also bridge the gap between complex machine learning models and non-technical business users.

# Hugging Face Space:

[Link to Interface](https://huggingface.co/spaces/DS5983-FACETS-team/FACETS-LLM-Assistant)

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
│   ├── EDA&Preprocessing.ipynb
│   └── customer_segmentation.ipynb
│
├── scripts/
│   ├── agent.py
│   ├── deepseek_agent.py
│   ├── ensemble_agent.py
│   ├── frontier_agent.py
│   ├── items.py
│   ├── pericer_service.py
│   ├── pericer_service2.py
│   ├── random_forest_agent.py
│   ├── retail_price_specialist_agent.py
│   ├── sales_specialist_agent.py
│   ├── segmentation_agent.py
│   ├── testing-2.py
│   ├── testing.py
│   └── varimax_agent.py
│
├── models/
│   ├── ensemble_model.pkl
│   ├── pretrained_timeseries_agent.pkl
│   ├── random_forest_model.pkl
│   └── varimax_agent.pkl
│
├── app.py
├── requirements.txt
├── environment.yml
├── README.md
├── .gitattributes
└── .gitignore

```
