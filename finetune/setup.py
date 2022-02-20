from setuptools import setup, find_packages

REQUIRED = [
    "transformers==4.16.2",
    "librosa==0.8.1",
    "numpy==1.22",
    "datasets==1.16.1",
    "jiwer==2.3.0"
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
    entry_points={
        'console_scripts':
            [
                'fine_tune_xlsr = finetune.finetune:run',
                'fine_tune_base = finetune.finetune_base:run',
             ]
    }
)
