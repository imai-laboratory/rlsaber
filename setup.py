#!/usr/bin/env python

import uuid
from setuptools import setup, find_packages
try:
    from pip._internal.req import parse_requirements
except ImportError:
    from pip.req import parse_requirements

#install_reqs = parse_requirements('requirements.txt', session=uuid.uuid1())

setup(name="rlsaber",
        version="0.1",
        license="MIT",
        description="Reinforcement learning tools like the galactic weapon",
        url="https://github.com/imai-laboratory/rlsaber",
        packages=find_packages())
