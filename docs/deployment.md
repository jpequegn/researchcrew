# ResearchCrew Deployment Guide

This guide covers deploying ResearchCrew to Vertex AI Agent Engine for production use.

## Prerequisites

1. **Google Cloud Platform Account** with billing enabled
2. **Google Cloud SDK** (`gcloud`) installed and configured
3. **Docker** installed and running
4. **ADK CLI** installed: `pip install google-adk`
5. **Required APIs enabled** in your GCP project:
   - Vertex AI API
   - Cloud Storage API
   - Secret Manager API
   - Container Registry API

## Quick Start

```bash
# Set your GCP project
export GCP_PROJECT_ID=your-project-id
export GOOGLE_API_KEY=your-api-key

# Build, push, and deploy
./deploy/deploy.sh all
```

## Deployment Options

### Option 1: Automated Deployment (Recommended)

Use the deployment script for a streamlined experience:

```bash
# Full deployment
./deploy/deploy.sh all

# Or step by step
./deploy/deploy.sh build    # Build Docker image
./deploy/deploy.sh push     # Push to GCR
./deploy/deploy.sh deploy   # Deploy to Vertex AI
```

### Option 2: Manual Deployment

#### Step 1: Build Docker Image

```bash
docker build -t researchcrew:latest .
```

#### Step 2: Push to Container Registry

```bash
# Configure Docker for GCR
gcloud auth configure-docker

# Tag and push
docker tag researchcrew:latest gcr.io/$GCP_PROJECT_ID/researchcrew:latest
docker push gcr.io/$GCP_PROJECT_ID/researchcrew:latest
```

#### Step 3: Deploy with ADK

```bash
adk deploy \
    --project=$GCP_PROJECT_ID \
    --region=us-central1 \
    --config=deploy/config.yaml
```

### Option 3: Local Development

Run the agent locally in Docker:

```bash
export GOOGLE_API_KEY=your-api-key
./deploy/deploy.sh local

# Or manually
docker run -p 8080:8080 \
    -e GOOGLE_API_KEY=$GOOGLE_API_KEY \
    researchcrew:latest
```

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GCP_PROJECT_ID` | Google Cloud project ID | Yes |
| `GCP_REGION` | Deployment region | No (default: us-central1) |
| `GOOGLE_API_KEY` | Google API key | Yes (runtime) |
| `LOG_LEVEL` | Logging level | No (default: INFO) |
| `DEFAULT_MODEL` | Primary LLM model | No (default: gemini-2.0-flash) |

### Secrets Management

Sensitive values should be stored in Secret Manager:

```bash
# Create a secret
echo -n "your-api-key" | gcloud secrets create researchcrew-google-api-key \
    --data-file=- \
    --project=$GCP_PROJECT_ID

# Grant access to the service account
gcloud secrets add-iam-policy-binding researchcrew-google-api-key \
    --member="serviceAccount:$SERVICE_ACCOUNT" \
    --role="roles/secretmanager.secretAccessor"
```

Reference secrets in `deploy/config.yaml`:

```yaml
secrets:
  GOOGLE_API_KEY: "researchcrew-google-api-key:latest"
```

### Resource Configuration

Edit `deploy/config.yaml` to adjust resources:

```yaml
resources:
  memory: 1024  # MB
  cpu: 1000     # millicores (1000 = 1 vCPU)

scaling:
  min_instances: 0  # Scale to zero when idle
  max_instances: 5
```

## Monitoring

### Health Checks

The agent exposes a health endpoint at `/health`:

```bash
curl http://localhost:8080/health
```

### Logs

View logs in Cloud Console or via CLI:

```bash
gcloud logging read "resource.type=vertex_ai_agent" \
    --project=$GCP_PROJECT_ID \
    --limit=50
```

### Metrics

Metrics are exposed in Prometheus format and integrated with Cloud Monitoring.

View in Cloud Console:
- **Metrics Explorer**: Custom metrics under `custom.googleapis.com`
- **Agent Dashboard**: Vertex AI Agent Engine monitoring

## Troubleshooting

### Common Issues

#### 1. Image Build Fails

```bash
# Check Docker is running
docker info

# Build with verbose output
docker build --progress=plain -t researchcrew:latest .
```

#### 2. Push Fails

```bash
# Ensure authenticated
gcloud auth configure-docker

# Check project access
gcloud projects describe $GCP_PROJECT_ID
```

#### 3. Deployment Fails

```bash
# Check ADK version
adk --version

# Validate config
./deploy/deploy.sh validate

# Check API is enabled
gcloud services list --enabled | grep -i vertex
```

#### 4. Runtime Errors

```bash
# Check logs
gcloud logging read "resource.type=vertex_ai_agent AND severity>=ERROR" \
    --project=$GCP_PROJECT_ID

# Verify secrets
gcloud secrets versions access latest --secret=researchcrew-google-api-key
```

### Debug Mode

Run with debug logging:

```bash
docker run -p 8080:8080 \
    -e GOOGLE_API_KEY=$GOOGLE_API_KEY \
    -e LOG_LEVEL=DEBUG \
    researchcrew:latest
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Deploy to Vertex AI

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Authenticate to GCP
        uses: google-github-actions/auth@v2
        with:
          credentials_json: ${{ secrets.GCP_SA_KEY }}

      - name: Set up Cloud SDK
        uses: google-github-actions/setup-gcloud@v2

      - name: Build and Push
        run: |
          gcloud auth configure-docker
          docker build -t gcr.io/${{ vars.GCP_PROJECT_ID }}/researchcrew:${{ github.sha }} .
          docker push gcr.io/${{ vars.GCP_PROJECT_ID }}/researchcrew:${{ github.sha }}

      - name: Deploy
        run: |
          pip install google-adk
          adk deploy --project=${{ vars.GCP_PROJECT_ID }} --region=us-central1
```

## Security Best Practices

1. **Never commit secrets** to version control
2. **Use Secret Manager** for all sensitive values
3. **Run as non-root** (configured in Dockerfile)
4. **Enable VPC** for private networking in production
5. **Restrict ingress** to internal or load balancer only
6. **Regular security scans** of container image

## Cost Optimization

1. **Scale to zero** when not in use (`min_instances: 0`)
2. **Right-size resources** based on actual usage
3. **Use committed use discounts** for predictable workloads
4. **Monitor and alert** on unexpected usage spikes

## Support

For issues with deployment:
1. Check the [troubleshooting section](#troubleshooting)
2. Review logs in Cloud Console
3. Open an issue in the repository
