import setuptools

setuptools.setup(
    name='min-dalle',
    description = 'min(DALL·E)',
    version='0.1.4',
    author='Brett Kuprel',
    author_email = 'brkuprel@gmail.com',
    packages=[
        'min_dalle', 
        'min_dalle.models'
    ],
    license='MIT',
    install_requires=[
        'torch>=1.11.0',
        'typing_extensions>=4.1.0'
    ],
    keywords = [
        'artificial intelligence',
        'deep learning',
        'text to image'
    ]
)