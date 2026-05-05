from setuptools import find_packages, setup

setup(
    name="egoemg_benchmark",
    version="0.1.0",
    description="EgoEmg: A Multimodal Egocentric Dataset with Bilateral EMG and Vision for Hand Pose Estimation",
    author="Anonymous Authors",
    author_email="anonymous@institution.edu",
    packages=find_packages(),
    install_requires=[
        # Left empty so you use the conda environment.yml file
    ],
)
