# Deploying to AWS EC2 with the model in S3

This guide deploys the serving API and dashboard to a single EC2 instance with
Docker Compose, and stores the versioned model artifact in a **private,
versioned, encrypted S3 bucket** that the instance downloads at startup.

> Principle: the EC2 instance gets an **IAM role** (never hard-coded keys) with
> **least-privilege** read access to just the model prefix. Secrets such as the
> PostgreSQL password live in `.env` on the instance, never in git.

---

## 0. Prerequisites

- An AWS account and a region (we use `us-east-1`).
- `aws` CLI v2 configured locally, `terraform` optional.
- Docker installed locally (to push) or build directly on the instance.

---

## 1. Create the S3 bucket (private, versioned, encrypted)

```bash
aws s3api create-bucket \
  --bucket ml-serving-drift-platform-models \
  --region us-east-1

# Block ALL public access.
aws s3api public-access-block \
  --bucket ml-serving-drift-platform-models \
  --public-access-block-configuration \
  BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true

# Enable object versioning (rollback + audit trail).
aws s3api put-bucket-versioning \
  --bucket ml-serving-drift-platform-models \
  --versioning-configuration Status=Enabled

# Server-side encryption with an AWS-managed key (SSE-S3).
aws s3api put-bucket-encryption \
  --bucket ml-serving-drift-platform-models \
  --server-encryption-configuration \
  '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}'
```

---

## 2. Create the IAM role and attach the least-privilege policy

`infra/s3_policy.json` grants `ListBucket` on the bucket and `GetObject`/`PutObject`
on the `models/*` prefix only.

```bash
# Trust policy allowing EC2 to assume the role.
cat > trust.json <<'JSON'
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Service": "ec2.amazonaws.com"},
    "Action": "sts:AssumeRole"
  }]
}
JSON

aws iam create-role --role-name ml-serving-drift-platform \
  --assume-role-policy-document file://trust.json

aws iam put-role-policy --role-name ml-serving-drift-platform \
  --policy-name s3-model-access \
  --policy-document file://infra/s3_policy.json

aws iam create-instance-profile \
  --instance-profile-name ml-serving-drift-platform

aws iam add-role-to-instance-profile \
  --instance-profile-name ml-serving-drift-platform \
  --role-name ml-serving-drift-platform
```

---

## 3. Train and upload the model to S3 (from your workstation)

```bash
export MODEL_STORE=s3 S3_BUCKET=ml-serving-drift-platform-models AWS_REGION=us-east-1
python -m pipeline.train --dataset adult
# This writes models/<UTC-version>/{model.joblib,metadata.json,baseline.csv} to S3.

aws s3 ls s3://ml-serving-drift-platform-models/models/ --recursive
```

---

## 4. Launch the EC2 instance

- AMI: **Amazon Linux 2023** or **Ubuntu 22.04**, `t2.small`/`t3.small` is enough for a demo.
- **Instance profile**: `ml-serving-drift-platform` (from step 2).
- **Security group** - open only what you need:

| Port | Source | Purpose |
|---|---|---|
| 22  | Your IP/32 | SSH (restrict to your CIDR) |
| 8000 | Your IP/32 or 0.0.0.0/0 | API (restrict unless you want it public) |
| 8501 | Your IP/32 | Streamlit dashboard |

Do **not** expose 5432 publicly - PostgreSQL stays on the Docker bridge network.

---

## 5. Install Docker and clone the repo (on the instance)

```bash
sudo yum update -y          # or `sudo apt update -y` on Ubuntu
sudo yum install -y docker git
sudo systemctl enable --now docker
sudo usermod -aG docker $USER   # re-login for group change

git clone <your-repo-url> ~/ml-serving-drift-platform
cd ~/ml-serving-drift-platform
```

---

## 6. Configure environment (no secrets in git)

```bash
cp .env.example .env
```

Edit `.env`:

```ini
MODEL_STORE=s3
S3_BUCKET=ml-serving-drift-platform-models
MODEL_VERSION=latest          # or pin a timestamp to roll back
AWS_REGION=us-east-1

POSTGRES_DB=mlmonitor
POSTGRES_USER=mluser
POSTGRES_PASSWORD=<strong-random-password>
```

Because the instance has the IAM role, boto3 picks up temporary credentials
automatically - **no AWS keys in `.env`**.

---

## 7. Start the stack

```bash
docker compose up -d --build
```

The API lifespan downloads the model from S3 at startup; PostgreSQL keeps
prediction logs on a named volume; the dashboard reads from PostgreSQL.

---

## 8. Verify

```bash
curl http://localhost:8000/health
# {"status":"ok","model_loaded":true,"model_version":"2026...Z","dataset":"adult"}

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features":{"age":42,"workclass":"Private","fnlwgt":180000,"education":"Some-college","education-num":9,"marital-status":"Married-civ-spouse","occupation":"Sales","relationship":"Husband","race":"White","sex":"Male","capital-gain":0,"capital-loss":0,"hours-per-week":45,"native-country":"United-States"}}'
```

From your workstation:

```bash
curl http://<EC2_PUBLIC_IP>:8000/health
# Dashboard: http://<EC2_PUBLIC_IP>:8501  (open only to your IP)
```

---

## 9. Roll back a model version

Edit `.env` and pin an older UTC version, then reload without a restart:

```bash
curl -X POST http://localhost:8000/reload   # picks up MODEL_VERSION from env
```

or set `MODEL_VERSION=20260101T093000Z` and `docker compose up -d api`.

---

## 10. Teardown (avoid orphaned charges)

```bash
aws ec2 terminate-instances --instance-ids <id>
aws s3 rm s3://ml-serving-drift-platform-models --recursive
aws s3api delete-bucket --bucket ml-serving-drift-platform-models
```
