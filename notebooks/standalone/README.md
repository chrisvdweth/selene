# Standalone Notebooks - Cloud-Ready

The original SELENE notebooks focus on clarity for better understanding and learning. This means that many notebooks import auxiliary code (e.g., for downloading files, plotting  graphs, checking for GPUs, etc.) that is used in the notebooks but is not essential and would clutter the notebook. On the flip side, this also means that you cannot simply upload most of the notebooks to a Cloud platform such as Google Colab as the platforms would not know where to find the auxiliary code.

In this folder, we therefore provide standalone versions of all the notebooks which are Cloud-ready, with all required auxiliary code and configuration data added to the notebook to remove any dependencies in terms of imports of auxiliary code. When using these standalone notebooks, please consider the following

* These standalone notebooks are **auto-generated** by converting the original notebooks using a custom script.

* Any auxiliary code and configuration data is prepended to the original notebook; you simply need to run each code cell before running the main content of the notebook.

* Depending on the Cloud platform you may get errors because required libraries are missing. In these cases, you will need to install libraries using, e.g., the `!pip` command directly in the notebooks; the notebooks shows a concrete example at the beginning.

* The standalone notebooks are not manually curated or (fully) tested. If you face issues that you cannot resolve (e.g., installing libraries), you can let us know and we will look into it.
