# Bocconi Admission Predictor

**Live Application:** [chance-me.com](https://chance-me.com)

A web application that estimates Bocconi University admission probabilities based on historical applicant data. The tool currently serves 100+ active users across 10+ countries.

## Architecture & Implementation

* **Analytical Engine:** Python backend implementing a custom K-nearest-neighbors (KNN) algorithm and percentile analysis. It evaluates SAT/GPA combinations against a curated dataset of 600+ historical profiles.
* **Frontend:** Lightweight HTML/Vanilla JS implementation for fast load times.
* **Infrastructure:** Deployed via Vercel using Serverless Functions (`/api` routing) for scalable backend execution without dedicated servers.

## Tech Stack

* Python (Data Analysis, KNN)
* HTML / CSS / JavaScript
* Vercel (Serverless Deployment)