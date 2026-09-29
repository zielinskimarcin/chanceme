# Bocconi Admission Predictor

**Live Application:** [chance-me.com](https://chance-me.com)

A web application that estimates Bocconi University admission probabilities based on historical applicant data. The calculator compares a user's scores with 600+ historical applicant profiles. Historical data is self-reported and the result is an estimate, not an official Bocconi admission decision.

## Architecture & Implementation

* **Analytical Engine:** Python backend implementing a custom K-nearest-neighbors (KNN) algorithm and percentile analysis. It evaluates SAT/GPA combinations against a curated dataset of 600+ historical profiles.
* **Frontend:** Lightweight HTML/Vanilla JS implementation for fast load times.
* **Infrastructure:** Deployed via Vercel using Serverless Functions (`/api` routing) for scalable backend execution without dedicated servers.

## Tech Stack
 
* Python (Data Analysis, KNN)
* HTML / CSS / JavaScript
* Vercel (Serverless Deployment)

## Local development

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r api/requirements.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python api/index.py
```

The Flask server exposes `/api/calculate` and `/api/health`. The static site contains the calculator homepage and source-backed admission guides such as `/bocconi-sat-score/`. To test the full application locally, serve the repository root and proxy `/api` to Flask, or use the Vercel development environment.

## Analytics and privacy

The production page currently loads GA4, Microsoft Clarity, and Vercel Insights. Product events intentionally exclude raw SAT/Bocconi Test scores and GPA. Before expanding tracking, review consent and retention requirements for the countries being served.
