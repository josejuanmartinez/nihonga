"""Build and check the replay environment in the NEW account, without a GPU."""
import hashlib
import json
from pathlib import Path
import sys

from modal_accounts import set_modal_account
from experiment_runtime import pin_reference_runtime


def check_environment(expected,source_hash):
    import hashlib
    import importlib.metadata
    from pathlib import Path
    import torch
    from diffusers.models.transformers import transformer_qwenimage21
    packages={name:importlib.metadata.version(name) for name in expected['packages']}
    actual_source=hashlib.sha256(Path(transformer_qwenimage21.__file__).read_bytes()).hexdigest()
    report=dict(account='new',gpu_requested=False,packages=packages,torch_cuda=torch.version.cuda,
                cudnn_version=torch.backends.cudnn.version(),transformer_source_sha256=actual_source)
    report['status']='passed' if (packages==expected['packages'] and torch.version.cuda==expected['torch_cuda']
        and report['cudnn_version']==expected['cudnn_version'] and actual_source==source_hash) else 'failed'
    return report


def main():
    sys.stdout.reconfigure(encoding='utf-8');sys.stderr.reconfigure(encoding='utf-8')
    set_modal_account('current',purpose='compute')
    import modal
    root=Path(__file__).resolve().parents[1]
    recovered=root/'data/recovered/modal'
    key='95fcd2cfae35cb7f124b8af2865c30f928cbd681a0d015df5e56e8ae5ba4fd78'
    request=json.loads((recovered/'phase7/blocks45_healing'/key/'request.json').read_bytes())['request']
    pin_reference_runtime(request,recovered)
    runtime=request['teacher']['runtime']
    image=(modal.Image.debian_slim(python_version=runtime['python_version']).apt_install('git')
           .pip_install(*request['execution_packages']).env(request['runtime_environment']))
    app=modal.App('nihonga-new-account-runtime-check')
    function=app.function(image=image,serialized=True,cpu=2,memory=8192,timeout=600)(check_environment)
    with modal.enable_output(),app.run(detach=True):
        report=function.remote(request['execution_reference_runtime'],request['transformer_source_sha256'])
    (root/'data/recovery/new-account-runtime-check.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if report['status']!='passed':raise SystemExit(1)


if __name__=='__main__':main()
