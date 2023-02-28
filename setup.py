from setuptools import setup

  
  
# specify requirements of your package here
REQUIREMENTS = ['numpy', 'pandas', 'seaborn', 'scipy', 'tqdm', 'pyarrow', 'scikit-learn', 'matplotlib', 'fastparquet']
  

  
# calling the setup function 
setup(name='feature_engg_cdao',
      version='0.0.1',
      description='feature engineering',
      url='https://github.com/crunchdao/feature-engineering',
      author='Utkarsh - Matteo',
      author_email='utkarshp1161@gmail.com',
      license='MIT',
      packages=['src'],
      install_requires=REQUIREMENTS,
      )