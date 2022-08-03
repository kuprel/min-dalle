import setuptools
# from pathlib import Path

setuptools.setup(
    name='min-dalle',
    description = 'min(DALL·E)',
    # long_description=(Path(__file__).parent / "README.rst").read_text(),
    version='0.4.11',
    author='Brett Kuprel',
    author_email='brkuprel@gmail.com',
    url='https://github.com/kuprel/min-dalle',
    packages=[
        'min_dalle', 
        'min_dalle.models'
    ],
    license='MIT',
    install_requires=[
        'torch>=1.11',
        'typing_extensions>=4.1',
        'numpy>=1.21',
        'pillow>=7.1',
        'requests>=2.23',
        'emoji'
    ],
    keywords = [
        'artificial intelligence',
        'deep learning',
        'text-to-image',
        'pytorch'
    ]
)