import sagemaker #Train and Deploy ML models on AWS

#Run PyTorch training jobs on SageMaker
from sagemaker.pytorch import PyTorch


# IAM role that gives SageMaker permission to access AWS resources (S3, logs, etc.)
role = "YOUR_SAGEMAKER_ROLE"

estimator = PyTorch(
    entry_point="train.py",
    role=role,
    instance_count=1,
    instance_type="ml.m5.large",
    framework_version="2.0.0"
)

# Start the training job on SageMaker
estimator.fit({
    
    # "train" is a channel name (you define it)
    # It points to S3 location where training data is stored
    "train": "s3://your-bucket/data/"
})