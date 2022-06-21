"""Setuptools of FPN_TensorFlow"""
from setuptools import setup, find_packages
import sys
import os

assert sys.version_info >= (3, 6), "Sorry, Python < 3.6 is not supported."


class InstallPrepare:
    """
    Parsing dependencies
    """

    def __init__(self):
        self.project = os.path.join(os.path.dirname(__file__), "FPN_TensorFlow")
        self._long_desc = os.path.join(self.project, "..", "README.md")
        self._owner = os.path.join(self.project, "..", "OWNERS")
        self._requirements = os.path.join(self.project, "..",
                                          "requirements.txt")

    @property
    def long_desc(self):
        if not os.path.isfile(self._long_desc):
            return ""
        with open(self._long_desc, "r", encoding="utf-8") as fh:
            long_desc = fh.read()
        return long_desc

    @property
    def version(self):
        default_version = "0.1"
        return default_version

    @property
    def owners(self):
        default_owner = "FPN_TensorFlow"
        if not os.path.isfile(self._owner):
            return default_owner
        with open(self._owner, "r", encoding="utf-8") as fh:
            check, approvers = False, set()
            for line in fh:
                if not line.strip():
                    continue
                if check:
                    approvers.add(line.strip().split()[-1])
                check = (line.startswith("approvers:") or
                         (line.startswith(" -") and check))
        return ",".join(approvers) or default_owner

    @property
    def basic_dependencies(self):
        return self._read_requirements(self._requirements)

    @staticmethod
    def _read_requirements(file_path, section="all"):
        print(f"Start to install requirements of {section} "
              f"in FPN_TensorFlow from {file_path}")
        if not os.path.isfile(file_path):
            return []
        with open(file_path, "r", encoding="utf-8") as f:
            install_requires = [p.strip() for p in f.readlines() if p.strip()]
        if section == "all":
            return list(filter(lambda x: not x.startswith("#"),
                               install_requires))
        section_start = False
        section_requires = []
        for p in install_requires:
            if section_start:
                if p.startswith("#"):
                    return section_requires
                section_requires.append(p)
            elif p.startswith(f"# {section}"):
                section_start = True
        return section_requires


_infos = InstallPrepare()

setup(
    name='FPN_TensorFlow',
    version=_infos.version,
    description="FPN TensorFlow Algorithm",
    packages=find_packages(exclude=["tests", "*.tests",
                                    "*.tests.*", "tests.*"]),
    author=_infos.owners,
    author_email="pujie2@huawei.com",
    maintainer=_infos.owners,
    maintainer_email="",
    include_package_data=True,
    python_requires=">=3.6",
    long_description=_infos.long_desc,
    long_description_content_type="text/markdown",
    license="Apache License 2.0",
    url="https://github.com/kubeedge-sedna/FPN_Tensorflow",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
    ],
    install_requires=_infos.basic_dependencies,
)
