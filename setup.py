from setuptools import setup

setup(name='mem_leak_detection',
      version='0.1',
      description='Detection of ongoing memory leak using trend analysis',
      url='',
      author='Anshul Jindal',
      author_email='anshul.jindal@tum.de',
      license='MIT',
      packages=['mem_leak_detection'],
      install_requires=[
          'pandas',
          'numpy',
          'scikit-learn',
          'scipy',
          'statsmodels',
      ],
      zip_safe=False)
