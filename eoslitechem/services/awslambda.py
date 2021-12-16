from botocore.exceptions import ClientError
import boto3
import re
import requests
import json

from ..production import LAMBDA_CONFIG


class AwsLambdaService(object):

    def __init__(self, model_id):
        self.model_id = model_id
        self.endpoint_url = self.describe()["EndpointUrl"]

    def generate_aws_compatible_string(self, *items, max_length=63):
        trimmed_items = [
            item[0][: item[1]] if type(item) == tuple else item for item in items
        ]
        items = [item[0] if type(item) == tuple else item for item in items]

        for i in range(len(trimmed_items)):
            if len("-".join(items)) <= max_length:
                break
            else:
                items[i] = trimmed_items[i]

        name = "-".join(items)
        if len(name) > max_length:
            raise Exception(
                "AWS resource name {} exceeds maximum length of {}".format(name, max_length)
            )
        invalid_chars = re.compile("[^a-zA-Z0-9-]|_")
        name = re.sub(invalid_chars, "-", name)
        return name

    def generate_lambda_resource_names(self, name):
        sam_template_name = self.generate_aws_compatible_string(f"{name}-template")
        deployment_stack_name = self.generate_aws_compatible_string(f"{name}-stack")
        # repo should be (?:[a-z0-9]+(?:[._-][a-z0-9]+)*/)*[a-z0-9]+(?:[._-][a-z0-9]+)*''
        repo_name = self.generate_aws_compatible_string(f"{name}-repo").lower()
        return sam_template_name, deployment_stack_name, repo_name
        
        
    def describe(self):
        # get data about cf stack
        _, stack_name, _ = self.generate_lambda_resource_names(self.model_id)
        lambda_config = LAMBDA_CONFIG
        cf_client = boto3.client("cloudformation", lambda_config["region"])
        try:
            stack_info = cf_client.describe_stacks(StackName=stack_name)
        except ClientError:
            print(f"Unable to find {self.model_id} in your cloudformation stack.")
            return

        info_json = {}
        stack_info = stack_info.get("Stacks")[0]
        keys = [
            "StackName",
            "StackId",
            "StackStatus",
        ]
        info_json = {k: v for k, v in stack_info.items() if k in keys}
        info_json["CreationTime"] = stack_info.get("CreationTime").strftime(
            "%m/%d/%Y, %H:%M:%S"
        )
        info_json["LastUpdatedTime"] = stack_info.get("LastUpdatedTime").strftime(
            "%m/%d/%Y, %H:%M:%S"
        )
        # get Endpoints
        outputs = stack_info.get("Outputs")
        outputs = {o["OutputKey"]: o["OutputValue"] for o in outputs}
        info_json.update(outputs)
        return info_json
    
    def post(self, data):
        
        # make it json serializable
        X = []
        for d in data:
            X += [[float(x) for x in d]]

        url = self.endpoint_url

        res = requests.post(
            "{0}/run".format(url),
            headers={"content-type": "application/json"},
            data=json.dumps(X)).text

        return res
