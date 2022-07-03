import setuptools
from pathlib import Path

requirements = Path("requirements.txt").read_text().splitlines()

setuptools.setup(
    name='min-dalle',
    description = 'min(DALL·E)',
    long_description=(Path(__file__).parent / "README.md").read_text(),
    version='0.2.10',
    author='Brett Kuprel',
    author_email='brkuprel@gmail.com',
    url='https://github.com/kuprel/min-dalle',
    packages=[
        'min_dalle', 
        'min_dalle.models'
    ],
    license='MIT',
    install_requires=[
        'torch>=1.10.0',
        'typing_extensions>=4.1.0'
    ] + requirements,
    keywords = [
        'artificial intelligence',
        'deep learning',
        'text-to-image',
        'pytorch'
    ]
)