import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tortoise-tts",
    packages=setuptools.find_packages(),
    version="3.0.0",
    author="James Betker",
    author_email="james@adamant.ai",
    description="A high quality multi-voice text-to-speech library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/neonbjb/tortoise-tts",
    project_urls={},
    scripts=[
        'scripts/tortoise_tts.py',
    ],
    include_package_data=True,
    install_requires=[
        'tqdm',
        'rotary_embedding_torch',
        'inflect',
        'progressbar',
        'einops',
        'unidecode',
        'scipy',
        'librosa',
        'transformers==4.29.2',
        'tokenizers==0.11.2',
        'scipy==1.13.1',
        'deepspeed==0.14.4',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
