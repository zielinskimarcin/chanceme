#!/bin/zsh
set -eu
project_root=${0:A:h:h}
cd "$project_root"
/Users/mz/.local/bin/vercel env run -e development --project bocconi-chanceme --scope zielinskimarcins-projects -- .venv/bin/python scripts/report_email_history.py
open "$project_root/qa/report-requests.html"
