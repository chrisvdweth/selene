# SELENE

## Announcements

* (20/12/2025) The new folder notebooks/standalone/ contains standalone versions of the original notebooks with all the code required to run each notebook "as is" on Cloud platforms such as Google Colab.
* (22/08/2025) Join our [Discord Server](https://discord.gg/nQMzt4QAM8) for latest updates, support, feedback, and more.


## What is SELENE?

The goal of SELENE is to be an open platform for self-paced learning and self-assessment, with a focus on but not limited to AI and related topics. Simply speaking, SELENE can be seen as a large-scale virtual textbook, going beyond the limitations of classic textbooks in the following ways:
* Unlimited scope w.r.t. to the topics covered (at least in principle)
* Support and emphasis on hands-on learning (*"Learning  by Doing"*) through interactive topic materials
* Easy linking between related and dependent topics to guide the learning process and provide mastery paths

The SELENE learning platform consists of two core components:
* **Jupyter notebook repository (here on Github):** Jupyter Notebooks are an open-source, interactive computing environment that allows you to create and share documents containing live code, equations, visualizations, and narrative text. They allow for a hands-on learning experience through modifying and running code examples. The repository will be a comprehensive and well-structured set of such notebooks organized into meaningful topics, projects, and applications.
* **Interactive web interface:** The web interface will allow users to visualize, navigate, and search for topics. It will also be able to recommend and visualize mastery paths for self-paced learning and provide knowledge checks for self-assessment. The web interface is currently under active development and will soon be available as a first prototype for testing. As a teaser, here are two early screenshots of the SELENE web interface:

<p align="center" width="100%">
 <img src='https://github.com/user-attachments/assets/06b68b36-e0df-41a6-ae55-3e6b6d897aa6' width='90%' /><br />
 <img src='https://github.com/user-attachments/assets/1536096e-a2af-403e-beb3-b2598014685c' width='90%' /><br />
</p>
<br />

## Troubleshooting

SELENE is a growing and evolving platform. While the notebooks can be viewed directly here in Github, their interactive use and exploration assumes users to download or clone the repository to locally run the notebooks in a [Jupyter](https://jupyter.org/) or similar environment on top of Python.

### Missing Libraries/Packages

Most notebooks import common Python packages such as `numpy`, `pandas`, `sklearn`, `networkx`, `torch` or others. All imports are done at the beginning of each notebook so you can see if you have all required packages locally installed. If you get any errors regarding missing packages, you simply need to installed using your package manager of choice  (e.g., [conda](https://anaconda.org/anaconda/conda) or [pip](https://pypi.org/project/pip/)).

### Errors Downloading Data

Many notebooks use datasets for interactive examples. Since these datasets can be very large, they are not part of the Github repository itself. Instead, all data files (e.g., datasets or pretrained models) are hosted on a separate server and are downloaded at the beginning of a notebook using auxiliary methods provided by SELENE. If downloading files fails, please check your local copy configuration file `config.yaml` if the URLs for downloading datasets, models, etc. are up to date and match the URLs in the copy of the configuration file on Github. If not, simply download or pull the latest version from Github.


## Licensing

### Content License
- Code in `src/`: [MIT License](LICENSE-MIT.md)
- Images in `images/`: [CC BY 4.0](LICENSE-CC-BY.md)

### Notebook License
- Code cells: [MIT License](LICENSE-MIT.md)
- Text/figures: [CC BY 4.0](LICENSE-CC-BY.md)

