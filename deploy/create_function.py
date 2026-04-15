import os
from pathlib import Path

import boto3
from botocore.exceptions import ClientError


ZIP_PATH = Path("word_count.zip")
FUNCTION_RUNTIME = "python3.12"
FUNCTION_HANDLER = "word_count.word_count_handler"


def get_required_env(name):
    """
    Read a required environment variable or fail with a clear message.
    """
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def load_zip_file(zip_path):
    """
    Load the deployment package from disk.
    """
    if not zip_path.exists():
        raise FileNotFoundError(f"Deployment package not found: {zip_path}")

    with zip_path.open("rb") as file_obj:
        return file_obj.read()


def lambda_function_exists(client, function_name):
    """
    Return True when the target Lambda function already exists.
    """
    try:
        client.get_function(FunctionName=function_name)
        return True
    except ClientError as exc:
        if exc.response["Error"]["Code"] == "ResourceNotFoundException":
            return False
        raise


def main():
    function_name = get_required_env("AWS_LAMBDA_FUNCTION_NAME")
    lambda_role_arn = get_required_env("AWS_LAMBDA_ROLE_ARN")
    zip_to_deploy = load_zip_file(ZIP_PATH)

    lambda_client = boto3.client(
        "lambda",
        aws_access_key_id=get_required_env("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=get_required_env("AWS_SECRET_ACCESS_KEY"),
        region_name=get_required_env("AWS_REGION"),
    )

    if lambda_function_exists(lambda_client, function_name):
        lambda_client.update_function_code(
            FunctionName=function_name,
            ZipFile=zip_to_deploy,
        )
        lambda_client.update_function_configuration(
            FunctionName=function_name,
            Role=lambda_role_arn,
            Runtime=FUNCTION_RUNTIME,
            Handler=FUNCTION_HANDLER,
        )
        print(f"Updated Lambda function: {function_name}")
        return

    lambda_client.create_function(
        FunctionName=function_name,
        Runtime=FUNCTION_RUNTIME,
        Role=lambda_role_arn,
        Handler=FUNCTION_HANDLER,
        Code={"ZipFile": zip_to_deploy},
    )
    print(f"Created Lambda function: {function_name}")


if __name__ == "__main__":
    main()
