from setuptools import setup

setup(name='spam',
      packages=['spam'],
      version='0.0.1dev1',
       entry_points={
            'console_scripts': ['spam-cli=spam.cmd:main']
      }
)
    