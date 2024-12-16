from setuptools import setup, find_packages

setup(
    name="mtbenchmark",          # 包名
    version="1.0.0",                       # 版本号
    author="AgentGuo",
    author_email="841796600@example.com",
    description="A time-series prediction benchmarking tool tailored to enterprise scenarios.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/AgentGuo/MTB",  # GitHub 地址
    packages=find_packages(),             # 自动发现模块
    install_requires=[                    # 依赖项
        "torch>=1.10",
        "numpy>=1.18",
        "pandas>=1.2",
        "gdown>=5.0"
    ],
    classifiers=[                         # 分类标签
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",              # Python 版本要求
)