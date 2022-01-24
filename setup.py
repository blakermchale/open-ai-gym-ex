from setuptools import setup

setup(name='open_ai_gym_ex',
    version='1.0.0',
    install_requires=[
        'numpy',
        'Pillow',
        'matplotlib',
        'cycler',
        'gym',
        'pybullet',
        'stable_baselines3',
        'ray[rllib]',
        'pandas'
        ]
)