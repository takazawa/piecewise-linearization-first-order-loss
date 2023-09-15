from setuptools import setup

packages = ["partition"]

package_data = {"": ["*"]}

install_requires = ["scipy>=1.8.1,<2.0.0"]

setup_kwargs = {
    "name": "partition",
    "version": "0.1.0",
    "description": "",
    "long_description": None,
    "author": "Yotaro Takazawa",
    "author_email": None,
    "maintainer": None,
    "maintainer_email": None,
    "url": None,
    "packages": packages,
    "package_data": package_data,
    "install_requires": install_requires,
    "python_requires": ">=3.8,<3.11",
}


setup(**setup_kwargs)
