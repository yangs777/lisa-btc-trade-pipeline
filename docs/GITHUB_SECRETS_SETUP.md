# GitHub Secrets Setup Guide

This guide explains how to set up GitHub Secrets for the CI/CD pipeline.

## Required Secrets

### 1. GCP_SERVICE_ACCOUNT_KEY

The base64-encoded content of your GCP service account JSON key file.

```bash
# Encode your service account key
base64 -w 0 < my-project-779482-xxxxx.json
```

Then add to GitHub:
1. Go to https://github.com/yangs777/lisa-btc-trade-pipeline/settings/secrets/actions
2. Click "New repository secret"
3. Name: `GCP_SERVICE_ACCOUNT_KEY`
4. Value: Paste the base64 output
5. Click "Add secret"

### 2. GCP_PROJECT_ID

- Name: `GCP_PROJECT_ID`
- Value: `my-project-779482`

### 3. GCS_BUCKET_NAME

- Name: `GCS_BUCKET_NAME`
- Value: `btc-orderbook-data`

## Available Service Accounts

We have 4 service accounts with different permissions:

1. **Vertex AI Runner** (`vertex-ai-runner-574@my-project-779482.iam.gserviceaccount.com`)
   - File: `my-project-779482-42a97ce77d1f.json`
   - Purpose: Running Vertex AI training pipelines

2. **App Engine Default** (`my-project-779482@appspot.gserviceaccount.com`)
   - File: `my-project-779482-5bd2bac5eab4.json`
   - Purpose: General application services

3. **Compute Engine Default** (`305343535055-compute@developer.gserviceaccount.com`)
   - File: `my-project-779482-7b22398028c4.json`
   - Purpose: VM and compute operations

4. **Drive Uploader** (`drive-uploader@my-project-779482.iam.gserviceaccount.com`)
   - File: `my-project-779482-c54c4ccb6bd6.json`
   - Purpose: Cloud Storage operations

## Local Development

For local development, set the environment variable:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/my-project-779482-c54c4ccb6bd6.json
```

Or add to `.env`:
```
GOOGLE_APPLICATION_CREDENTIALS=/path/to/my-project-779482-c54c4ccb6bd6.json
```

## Verifying Setup

After setting up secrets, the CI pipeline will automatically:
1. Authenticate with GCP
2. Verify bucket access
3. Run all tests with proper credentials

You can manually trigger a CI run to verify everything is working.