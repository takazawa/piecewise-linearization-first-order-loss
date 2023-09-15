# -*- coding: utf-8 -*-
from setuptools import setup

packages = ["sample_lib"]

package_data = {"": ["*"]}

setup_kwargs = {
    "name": "sample-lib",
    "version": "0.1.0",
    "description": "",
    "long_description": None,
    "author": "Author name",
    "author_email": None,
    "maintainer": None,
    "maintainer_email": None,
    "url": None,
    "packages": packages,
    "package_data": package_data,
    "python_requires": ">=3.8,<3.11",
}

setup(**setup_kwargs)
