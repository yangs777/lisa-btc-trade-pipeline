@echo off
docker build -t gcr.io/my-project-779482/lisa-btc-trade:latest .
gcloud auth configure-docker -q
docker push gcr.io/my-project-779482/lisa-btc-trade:latest
pause