#!/bin/bash
# Quick start script for Real-Time Multi-API Data Dashboard
cd "$(dirname "$0")"
streamlit run src/dashboard.py --server.headless true
