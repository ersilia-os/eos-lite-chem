import os
import subprocess
import tempfile
import shutil
import json

from . import LAMBDA_CONFIG


BENTOML_AWS_LAMBDA_TOOL = "aws-lambda-deploy"
LAMBDA_CONFIG_JSON = "lambda_config.json"


class AwsLambdaDeployer(object):
    def __init__(self, model_id):
        self.model_id = model_id

    def check_docker(self):
        cmd = "docker run hello-world"

    def _download_bentoml_aws_lambda_tool(self):
        tool_dir = tempfile.mkdtemp(prefix="eos-lite-chem")
        cwd = os.getcwd()
        os.chdir(tool_dir)
        cmd = "git clone git@github.com:bentoml/{0}.git".format(BENTOML_AWS_LAMBDA_TOOL)
        subprocess.Popen(cmd, shell=True).wait()
        os.chdir(cwd)
        self._replace_lambda_config(tool_dir)
        return tool_dir

    def _replace_lambda_config(self, tool_dir):
        json_file = os.path.join(tool_dir, BENTOML_AWS_LAMBDA_TOOL, LAMBDA_CONFIG_JSON)
        with open(json_file, "w") as f:
            json.dump(LAMBDA_CONFIG, f, indent=4)

    def _delete_aws_lambda_tool(self, tool_dir):
        shutil.rmtree(tool_dir)

    def _get_bundle_path(self):
        cmd = "bentoml get OnnxInference:latest --print-location -q"
        tmp_dir = tempfile.mkdtemp()
        bundle_file = "bundle.txt"
        with open(os.path.join(tmp_dir, bundle_file), "w") as f:
            subprocess.Popen(cmd, stdout=f, shell=True).wait()
        with open(os.path.join(tmp_dir, bundle_file), "r") as f:
            for l in f:
                bundle = l.rstrip(os.linesep)
        shutil.rmtree(tmp_dir)
        return bundle

    def _deploy_bundle(self, tool_dir):
        deploy_script = os.path.join(tool_dir, BENTOML_AWS_LAMBDA_TOOL, "deploy.py")
        bento_bundle_path = self._get_bundle_path()
        config_file = os.path.join(
            tool_dir, BENTOML_AWS_LAMBDA_TOOL, LAMBDA_CONFIG_JSON
        )
        cmd = "python {0} {1} {2} {3}".format(
            deploy_script, bento_bundle_path, self.model_id, config_file
        )
        subprocess.Popen(cmd, shell=True).wait()

    def _describe(self, tool_dir):
        describe_script = os.path.join(tool_dir, BENTOML_AWS_LAMBDA_TOOL, "describe.py")
        config_file = os.path.join(
            tool_dir, BENTOML_AWS_LAMBDA_TOOL, LAMBDA_CONFIG_JSON
        )
        cmd = "python {0} {1} {2}".format(describe_script, self.model_id, config_file)
        subprocess.Popen(cmd, shell=True).wait()

    def _delete_local_deployment_folder(self):
        for n in os.listdir():
            if "-lambda-deployable" in n:
                print("Deleting lambda deployable")
                shutil.rmtree(n)

    def deploy(self):
        tool_dir = self._download_bentoml_aws_lambda_tool()
        self._deploy_bundle(tool_dir)
        self._describe(tool_dir)
        self._delete_aws_lambda_tool(tool_dir)
        self._delete_local_deployment_folder()
