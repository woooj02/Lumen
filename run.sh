#!/usr/bin/env bash
# Launch Lumen. Unsets PYTHONPATH so a system (e.g. ROS) PYTHONPATH can't shadow
# the venv's packages. Uses `python -m streamlit` so it works even if the venv
# was moved. Headless + no usage prompt so it boots straight to a serving URL.
cd "$(dirname "$0")"
env -u PYTHONPATH ./venv/bin/python -m streamlit run app.py \
  --server.headless true \
  --server.port 8501 \
  --browser.gatherUsageStats false
