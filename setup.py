from setuptools import setup, find_packages

setup(
  name="Readout_Error_Mitigation",
  version="0.1",
  author="Your Name",
  author_email="your.email@example.com",
  description="A library for readout error mitigation",
  long_description=open('README.md').read(),
  long_description_content_type='text/markdown',
  url="https://github.com/yifan1207/Readout_Error_Mitigation",
  packages=find_packages(where='src'),
  package_dir={'': 'src'},
  classifiers=[
  "Programming Language :: Python :: 3",
  "License :: OSI Approved :: MIT License",
  "Operating System :: OS Independent",
  ],
  python_requires='>=3.6',
  install_requires=[
  # List your package dependencies here
  # Example: "numpy >= 1.19.2",
  ],
  tests_require=[
  'pytest',
  ],
)
