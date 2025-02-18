# Default Data Folder

Many notebooks require additional data (e.g., datasets or pretrained models) to run. However, since datasets or pretrained models can be very large, none of those data files are part of the Github repository. All files a notebook requires are downloaded within the notebook itself.

This `data/` folder is the default location for any downloaded files. If you want to set your own target location you can specify it using the `download_path` parameter of the utility methods such as `download_dataset()` or `download_model()` used in the notebooks to download the files.
