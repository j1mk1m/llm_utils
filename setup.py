from setuptools import setup, find_packages

setup(
    name='llm_utils',
    version='1.0',
    author='James Kim',
    author_email='j1mk1m1016@gmail.com',
    description='LLM Utilities',
    url='https://github.com/j1mk1m/llm_utils',
    packages=find_packages(),
    install_requires=[
        'litellm',
    ],
)