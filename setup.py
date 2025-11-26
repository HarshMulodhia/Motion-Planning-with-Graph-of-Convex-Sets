from setuptools import setup, find_packages

setup(
    name="gcs_grasp_rl",
    version="0.1",
    packages=find_packages(),  # Finds 'src' and 'scripts'
    install_requires=[
        "numpy",
        "pybullet",
        "stable-baselines3",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "opencv-python"
    ],
)
