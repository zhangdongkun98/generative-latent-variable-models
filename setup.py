from setuptools import setup, find_packages

setup(
    name='glvm',
    packages=find_packages(),
    version='0.0.1',
    author='Zhang Dongkun',
    author_email='zhangdongkun98@gmail.com',
    url='https://github.com/zhangdongkun98/generative-latent-variable-models',
    description='generative latent variable models',
    install_requires=[
        'rllib',
    ],

    include_package_data=True
)
