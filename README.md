# Garzoni Dataset Exploration #

This is the repository to explore the Garzoni Dataset. It will include the
following items:

- Code to work with the dataset
- Reports and findings
- Possible extra comments

## Initial Reporting ##

For the initial reporting, pandas data frames are used with an extension
library to profile the dataset. The extension library gives various details
about each variable in a data frame. These details can be viewed in the reports
folder. After the initial profiling of each data set, it's been noticed that
some variables are not profiled. Further examination revealed that these
variables are dictionaries in the data frame, which are mostly connections to
other data frames. These are the variables (The format is "data frame": "List
of dictionary variables"): 

- persons: relationships
- contracts: mentions
- person mentions: name, entity, professions, workshop, geological Origin,
  charge location, residence

Among these variables, only the two called 'workshop' and 'name' are considered
to hold extra information, so they have been turned into separate data frames
and reported respectively.
