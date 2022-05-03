import boto3

def upload_model():
    model_name = "models/best-checkpoint-v6.ckpt"
    key = "yohan-model-onnx"
    s3 = boto3.client(
        's3',  # 사용할 서비스 이름, ec2이면 'ec2', s3이면 's3', dynamodb이면 'dynamodb'
        aws_access_key_id="AKIA5GLHM4NCGJVBBMLW",         # 액세스 ID
        aws_secret_access_key="OHV7JFXPoId3QNjHZGnR/I0HZO0CN9Lm1HbYvnta"
    )  
    res = s3.upload_file(model_name, "mlops-study-project", key)
    print(res)

if __name__ == "__main__":
    upload_model()