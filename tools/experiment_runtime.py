"""Resolve previously unpinned execution dependencies against the saved reference."""
import json


def pin_reference_runtime(request,recovered_root):
    reference=json.loads((recovered_root/'phase6/deterministic_reference'/
        '434811953bbe987d418bdf26a191386706920405e535a750d654a194ecba8da8'/'summary.json').read_bytes())
    request['execution_packages']=[('pillow=='+reference['packages']['pillow']) if item.lower()=='pillow' else item
                                   for item in request['teacher']['runtime']['packages']]
    request['execution_reference_runtime']={key:reference[key] for key in ('packages','torch_cuda','cudnn_version')}
