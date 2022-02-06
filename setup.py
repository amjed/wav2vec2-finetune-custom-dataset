from setuptools import setup, find_packages

REQUIRED = [
    "torch==1.10.1",
    "transformers==4.13.0",
    "librosa==0.8.1"
    ]

setup(
    name='finetune',
    version='0.0.1',
    packages=find_packages(),
    author='ao',
    url='https://mindput.cc',
    author_email='amjed1@live.com',
    python_requires='>=3.6',
    include_package_data=True,
    install_requires=REQUIRED,
)
