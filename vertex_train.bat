@echo off
gcloud ai custom-jobs create ^
  --region=asia-northeast3 ^
  --display-name=lisa-btc-trade-rl ^
  --worker-pool-spec=machine-type=n1-standard-8,accelerator-type=NVIDIA_TESLA_T4,accelerator-count=1,executor-image-uri=gcr.io/my-project-779482/lisa-btc-trade:latest
pause