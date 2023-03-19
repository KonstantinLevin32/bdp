import os.path as osp

import setuptools

# cur_dir = osp.dirname(osp.realpath(__file__))
# requirementPath = osp.join(cur_dir, "requirements.txt")
# install_requires = []
# if osp.isfile(requirementPath):
#    with open(requirementPath) as f:
#        install_requires = f.read().splitlines()

setuptools.setup(
    name="bdp",
    version="0.1",
    author="",
    author_email="",
    description="",
    url="",
    # install_requires=install_requires,
    packages=setuptools.find_packages(),
)
